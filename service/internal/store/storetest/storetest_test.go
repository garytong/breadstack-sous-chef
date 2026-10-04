package storetest

import "testing"

func TestWithDatabase(t *testing.T) {
	cases := []struct{ in, name, want string }{
		{"postgres://u:p@h:5432/postgres?sslmode=disable", "t_ab", "postgres://u:p@h:5432/t_ab?sslmode=disable"},
		{"host=/cloudsql/x dbname=postgres user=u", "t_ab", "host=/cloudsql/x user=u dbname=t_ab"},
		{"host=h user=u", "t_ab", "host=h user=u dbname=t_ab"},
	}
	for _, c := range cases {
		if got := withDatabase(c.in, c.name); got != c.want {
			t.Errorf("withDatabase(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestRandomHexLength(t *testing.T) {
	if got := randomHex(6); len(got) != 12 {
		t.Errorf("randomHex(6) = %q (len %d), want 12 hex chars", got, len(got))
	}
}
