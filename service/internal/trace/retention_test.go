package trace

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/store"
)

func env(m map[string]string) Getenv { return func(k string) string { return m[k] } }

func TestPolicyFromEnv(t *testing.T) {
	p, err := PolicyFromEnv(env(nil))
	if err != nil || p != store.DefaultRetention {
		t.Fatalf("default = %+v, %v", p, err)
	}
	p, err = PolicyFromEnv(env(map[string]string{"TRACE_RETENTION_DAYS": " 30 "}))
	if err != nil || p.TraceDays != 30 || p.ConversationDays != 30 || p.ShippedUsageDays != 30 || p.InactivityDays != 30 {
		t.Fatalf("30 = %+v, %v", p, err)
	}
	for _, bad := range []string{"0", "-5", "abc", "1.5"} {
		if _, err := PolicyFromEnv(env(map[string]string{"TRACE_RETENTION_DAYS": bad})); err == nil {
			t.Errorf("TRACE_RETENTION_DAYS=%q accepted", bad)
		}
	}
}

type fakeRetainer struct {
	got store.RetentionPolicy
	res store.RetentionResult
	err error
}

func (f *fakeRetainer) Retention(_ context.Context, p store.RetentionPolicy) (store.RetentionResult, error) {
	f.got = p
	return f.res, f.err
}

func TestRun(t *testing.T) {
	var logs []string
	logf := func(f string, a ...any) { logs = append(logs, strings.TrimSpace(f)) }
	fr := &fakeRetainer{res: store.RetentionResult{Traces: 3, Conversations: 2, UsageRows: 1}}
	res, err := Run(context.Background(), fr, store.DefaultRetention, logf)
	if err != nil || res.Traces != 3 || fr.got != store.DefaultRetention || len(logs) != 1 || !strings.HasPrefix(logs[0], "retention: ok") {
		t.Fatalf("Run = %+v, %v, logs=%v", res, err, logs)
	}
	fr = &fakeRetainer{err: errors.New("boom")}
	logs = nil
	if _, err := Run(context.Background(), fr, store.DefaultRetention, logf); err == nil || !strings.HasPrefix(logs[0], "retention: FAILED") {
		t.Fatalf("failure not reported: %v %v", err, logs)
	}
	// nil logf falls back to log.Printf without panicking.
	if _, err := Run(context.Background(), &fakeRetainer{}, store.DefaultRetention, nil); err != nil {
		t.Fatal(err)
	}
}

func TestCommandDryRunAndFlags(t *testing.T) {
	var out bytes.Buffer
	err := Command(context.Background(), []string{"--dry-run", "--days", "7"}, env(nil), &out)
	if err != nil || !strings.Contains(out.String(), "retention dry-run") {
		t.Fatalf("dry-run: %v %q", err, out.String())
	}
	if err := Command(context.Background(), []string{"--days", "0", "--dry-run"}, env(nil), &out); err == nil {
		t.Error("--days 0 accepted")
	}
	if err := Command(context.Background(), []string{"--bogus"}, env(nil), &out); err == nil {
		t.Error("unknown flag accepted")
	}
	if err := Command(context.Background(), nil, env(map[string]string{"TRACE_RETENTION_DAYS": "x"}), &out); err == nil {
		t.Error("bad env accepted")
	}
	// Without DATABASE_URL a real run fails before connecting.
	if err := Command(context.Background(), nil, env(nil), &out); err == nil || !strings.Contains(err.Error(), "DATABASE_URL") {
		t.Errorf("missing DATABASE_URL: %v", err)
	}
}
