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

func TestRegistrationSource(t *testing.T) {
	cases := map[string]string{
		"":              SourceLab,
		"   ":           SourceLab,
		"lab":           SourceLab,
		"lab-r4":        SourceLab,
		"lab-exam":      SourceLab,
		"lab_x":         SourceLab,
		"lab.1":         SourceLab,
		"lab:2":         SourceLab,
		" LAB-R4 ":      SourceLab,
		"dev":           SourceLab, // a local launcher run (version.go default)
		"Dev":           SourceLab,
		"label":         SourceLauncher, // not lab + separator
		"labr4":         SourceLauncher,
		"1a2b3c4":       SourceLauncher,
		"v3.0.91":       SourceLauncher,
		"int-20261010a": SourceLauncher,
		"developer":     SourceLauncher,
	}
	for in, want := range cases {
		if got := RegistrationSource(in); got != want {
			t.Errorf("RegistrationSource(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestRetentionRegistrationCutoff(t *testing.T) {
	now := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	if got := (RetentionPolicy{}).RegistrationCutoff(now); !got.Equal(now.AddDate(0, 0, -DefaultRegistrationDays)) {
		t.Errorf("zero RegistrationDays cutoff = %s, want the 180-day default", got)
	}
	if got := (RetentionPolicy{RegistrationDays: 10}).RegistrationCutoff(now); !got.Equal(now.AddDate(0, 0, -10)) {
		t.Errorf("10-day cutoff = %s", got)
	}
	p := DefaultRetention
	p.RegistrationDays = -1
	if p.Validate() == nil {
		t.Error("negative RegistrationDays accepted")
	}
	p.RegistrationDays = 0
	if err := p.Validate(); err != nil {
		t.Errorf("RegistrationDays 0 (default) rejected: %v", err)
	}
}
