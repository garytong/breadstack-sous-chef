package prompt

import (
	"strings"
	"testing"
)

// The index's "Also called" lines carry the merchant vocabulary's container
// aliases (internal/fields vocab.txt) for the containers the manifest has.
func TestRenderIndex_AlsoCalled(t *testing.T) {
	idx, err := RenderIndex(loadSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	_, section, ok := strings.Cut(idx.Text, "\n## Also called (merchant words → container)\n")
	if !ok {
		t.Fatalf("no Also called section\n%s", idx.Text)
	}
	section, _, _ = strings.Cut(section, "\n## ")
	for _, want := range []string{
		"g:BrandingSection: site name, store name, shop name\n",
		"g:HeaderSection: announcement bar, ",
		"g:NotificationSection: cart popup, ",
		"b:branding: age gate, store picker, ",
	} {
		if !strings.Contains(section, want) {
			t.Errorf("Also called lacks %q\n%s", want, section)
		}
	}
	for _, not := range []string{
		"s:hero:",    // an aliased section type prints under heroBanner only
		"page:",      // page-view entries are retrieval-only
		"top banner", // an extension of "banner", which heroBanner's id says
		"announcement bar, top bar, navigation bar, promo bar", // capped at alsoCalledMax
	} {
		if strings.Contains(section, not) {
			t.Errorf("Also called has %q\n%s", not, section)
		}
	}
	// A manifest without the containers prints none of their lines.
	small, err := RenderIndex([]byte(smallManifest))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(small.Text, "g:BrandingSection:") {
		t.Errorf("small manifest lists an absent panel's aliases\n%s", small.Text)
	}
}
