package store

import (
	"testing"
	"time"
)

// Pure tests: no database.

func TestBuildLine(t *testing.T) {
	cases := map[string]string{
		"v3.0.91":  "prod",
		" v3.0.91": "prod",
		"abc1234":  "int",
		"dev":      "int",
		"":         "int",
		"V3.0.91":  "int", // launcher tags are lower-case v; anything else is int
	}
	for in, want := range cases {
		if got := BuildLine(in); got != want {
			t.Errorf("BuildLine(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestRetentionPolicyCutoffs(t *testing.T) {
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	tr, cv, us, idle := DefaultRetention.Cutoffs(now)
	if want := now.AddDate(0, 0, -90); !tr.Equal(want) {
		t.Errorf("trace cutoff %v want %v", tr, want)
	}
	if want := now.AddDate(0, 0, -90); !cv.Equal(want) {
		t.Errorf("conversation cutoff %v want %v", cv, want)
	}
	if want := now.AddDate(0, 0, -30); !us.Equal(want) {
		t.Errorf("usage cutoff %v want %v", us, want)
	}
	if want := now.AddDate(0, 0, -90); !idle.Equal(want) {
		t.Errorf("inactivity cutoff %v want %v", idle, want)
	}
}

func TestRetentionPolicyValidate(t *testing.T) {
	if err := DefaultRetention.Validate(); err != nil {
		t.Fatalf("default policy invalid: %v", err)
	}
	for _, p := range []RetentionPolicy{
		{TraceDays: 0, ConversationDays: 90, ShippedUsageDays: 30},
		{TraceDays: 90, ConversationDays: -1, ShippedUsageDays: 30},
		{TraceDays: 90, ConversationDays: 90, ShippedUsageDays: 0},
		{TraceDays: 90, ConversationDays: 90, ShippedUsageDays: 30, InactivityDays: 0},
	} {
		if err := p.Validate(); err == nil {
			t.Errorf("policy %+v should be invalid", p)
		}
	}
}

func TestNullableJSON(t *testing.T) {
	if nullableJSON(nil) != nil {
		t.Error("nil RawMessage must map to SQL NULL")
	}
	if nullableJSON([]byte{}) != nil {
		t.Error("empty RawMessage must map to SQL NULL")
	}
	if v := nullableJSON([]byte(`{}`)); v == nil {
		t.Error("non-empty RawMessage must pass through")
	}
}
