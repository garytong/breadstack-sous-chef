# Merchant vocabulary

`service/internal/fields/vocab.txt` holds the words merchants use where the editor's labels and keys use different ones. For example, a merchant says "site name" for the `storeName` field ("Store Name"), "rounder" for any `*Radius` field, "the banner" for the hero, and "hide the THC pill" for `cardShowTHC`.

Two parts of the service read this file:

- **The candidate-fields block** reads it on every turn (`internal/fields`, README §3).
- **The prompt index** prints its container aliases in the "Also called" lines of `system[1]` (`internal/prompt`).

The file is embedded in the binary, so a change takes effect on the next deploy. Nothing else needs to change.

## When a developer adds a field

Usually nothing needs to be done. Terms map merchant words to *label and key words*, not to fields, so a new field labelled "Card Border Radius" is already found by "rounder cards". Add a line only when merchants will call the new thing by a word that is in neither its label nor its key.

## Adding a term (one line)

```
merchant words = editor words[, editor words …]   # why: the request that needed it
```

For example:

```
rounder = radius                 # "make the product cards rounder"
site name = store name           # "change the site name to HashiTong"
```

- **Right-hand side.** Write words that occur in field labels or keys. Each one becomes an extra reading of the text at 0.8 of a written word's weight; the merchant's own words stay.
- **How words are compared.** Lower-cased, singular, after the spelling folds in `tokens.go` (colour→color, bg→background, btn→button, photo/picture/pic→image). Write the singular form.
- **Stop words** (`the`, `to`, `on`, `my` …):
  - A stop word on the left-hand side must appear in the text as written, so `turn on` does not match a bare "turn".
  - Any other stop word in the text is skipped, so `rename site` matches "rename my site".
- **Values never match.** A term never applies inside a value run (`… to 'Site name'`).
- **Keep the left side specific.** A reading competes with every label that has the word. `add = show` read every `show*` toggle into "add our pinterest pinterest.com/…" and they outscored the rare word the label spells ("Pinterest URL"); `add a = show` / `add an = show` still pulled them over "add a paper texture behind everything" (holdout h010, found at HEAD), so the lines are now the toggle phrases only: `add a second`, `add a search`, `add a promo` = show ("add a search box" still reads "Show search box"; a multi-word left side reads the words it covers as one reading). Check a new term against a request that spells a rare label next to it.
- **A term that names a container through a common word is a hint, not a name.** `font = typography` reads "font" as the Typography panel's id word. Because "font" is a field word of two or more other targets, the panel is then listed right after the targets the text names (like the open panel, README §3), not above them: "i don't like the font" lists Typography first, while "the Button Font on the buttons" stays the Buttons panel's and "the Footnote Font of contentBlock-2" does not list Typography at all (the word lies inside a longer label the block spells). A term from a word only its own container uses (`facebook = social`) still names it (`fields.go` `hintOnly`).

## Adding a container alias

```
@<kind>:<id> = phrase, phrase …    # kinds: s:<section type>, g:<GlobalPanel>, b:branding, p:<page>
```

An alias *names* the container, and a named target goes to the top of the block. So use a phrase only one container answers to. Avoid a label that another container also has: `link color` would pull the Typography panel above the shop menu's own "Link Color".

Never a bare word that other labels contain. `@p:seo` once had `google`, so "link our google play app" and "add our google business link" listed only the seo page and lost the Footer's "Show Google Play badge" and the social panel's "Google Business URL" (refute of 2026-10-10). It now lists phrases that mean the search listing (`google search`, `title on google`, `shows up on google` …). Page aliases drop their stop words, so "title on google" is the phrase "title google".

- **Page aliases** match word for word.
- **`page:<slug>`** names the container while the merchant views that page. For example, `@b:branding = …, page:age-gate` applies because that page holds only `published`.
- **Order matters.** The index prints the first three aliases its id does not already say.

## Checking a change

- `cd service && go test ./internal/fields ./internal/prompt` runs the parse tests, checks that every `@container` exists in the snapshot manifest, and runs the vocabulary tests (`vocab_test.go`: site name, rounder, instagram, announcement bar, heading font, hide/show polarity, traps).
- A container-alias edit that changes a printed "Also called" line moves the bytes of `system[1]`. That fails `TestFieldsContext_AnthropicRequestIsByteIdentical` (`internal/loop`); update its `anthropicBodySHA` once you have checked that only the index moved. Term lines never touch `system[1]`.
- `TestGolden` pins the block bytes for the lab's training rows. A vocabulary change that moves a golden is a retraining event: accept it with `-update` only on purpose.
- To measure recall, run `TestRecallOffline` (FIELDS_RECALL_CASES / FIELDS_RECALL_MANIFEST) on the lab's verbatim cases. Its recall must stay at 100% and its distractor rate must not rise. Then run the real-phrasing exam offline, `make -C lab exam-retrieval SET=all` (docs/LAB.md §6b; `WITH_SELECTED=1` and `FOLLOWUPS=1` for the focus and follow-up paths), and report `holdout` apart from the rows you tuned on.
- `lookup_fields` reads the vocabulary too: the server tool adds the terms' readings of the model's query (`fields.QueryReadings` → `manifest.LookupRequest.Readings`), matched after the query's own matches, so `query: "site name"` finds `storeName`.
