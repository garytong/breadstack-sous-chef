"""askrule — when a merchant's turn may be answered with a question (r7 spec §1.1).

One rule for every layer that labels `ask` (the authoring validation, contrast's ask-first conversion and vague
family, prepare's pool asks, the dataset's last check of every kept ask row — carried rows included):

    a turn is VAGUE iff it has no VALUE, no DIRECTION and no ACTION.  Only a vague turn may be labelled ask / offer.

It is the same thing service/internal/prompt/rules.go:94 already tells the teacher ("ask only when the request gives
no value, no direction and no change to make"); r6's labels broke it (15/48 relative exam requests asked), so the lab
now checks its own labels against it. `rules.go` is not touched.

  VALUE      V1 a colour literal (#hex, rgb( / hsl(, a colour word not after "too"); V2 a number, optionally with a
             unit (not an instance number: "panel 2", "button 2"); V3 a quoted string that is not the target's current
             text or a label (`names`: quoting 'Shop now' names the button); V4 a URL, domain, e-mail, @handle or
             /path; V5 to / into / as / should be / should read / should say / is / = / : followed by a capitalised
             token, a token with a digit or a quote, "rename … to X", say(s) / read(s) / call(ed) / name(d) / titled X
             (X not something / anything / else / different / new, and not a name the site already shows); V6 an
             option of the case's fields (`values`: enum values and labels, the manifest's fonts for a font field,
             asset names for an image field, the store's categories / tags / brands / locations for a reference
             field, page slugs and nav names for a link field); V7 a literal the source case's oracle writes (in
             `values` too: values_for(oracle_ops=…))
  DIRECTION  D1 a comparative (bigger, smaller, darker, lighter, wider …); D2 more / less / fewer (not "more info",
             "read more"); D3 too <adj>, not <adj> enough; D4 a verb that carries a direction (increase, reduce,
             shrink, darken, bump up, tone down, dial back …)
  ACTION     a determinate operation: hide / show (as the verb) / remove / delete / turn on-off / enable / reset /
             centre / align / uppercase / bold / sticky / move up-down / swap X and Y / open in a new tab / full
             width; add / include / put only with a section type name or a toggle's label; an image request (image,
             photo, picture, logo, icon, favicon, banner with change / replace / upload / new / different / swap) is
             the action pick_asset (rules.go:49: pick_asset lets the merchant choose)

Deliberately NOT a direction: "hard to read / see", "washed out" (darker or lighter depends on the background), and
the vague templates' own words (needs work, doesn't feel right, boring, out of date, change, update, fix, tweak).

Target ambiguity ("which section?") is a different, legitimate ask: mode clarify with candidates, or a case with
meta.ask_reason = "which_one". It is never inferred from an ask label, and this module does not judge it.

    classify(text, values=…, bool_labels=…, section_names=…, names=…) -> Verdict(vague, kind, why)
    direction_of(text, field_type) -> "up" | "down" | "darker" | "lighter" | None
    values_for(case, m, oracle_ops=…) -> the case's V6 + V7 vocabulary
    classify_case(case, m, text=…, oracle_ops=…) -> Verdict with the case's whole vocabulary
"""
from __future__ import annotations

import re
from typing import Any, Iterable, NamedTuple

GENERATION = "askrule-r7"

COLOUR_WORDS = (
    "red orange yellow green blue purple violet pink black white grey gray brown teal navy maroon gold silver beige "
    "cream ivory tan olive lime mint turquoise cyan magenta lavender coral peach salmon charcoal burgundy khaki indigo "
    "crimson scarlet emerald sage plum mustard aqua rust taupe").split()
COMPARATIVES = (
    "bigger larger smaller tinier taller shorter wider narrower thinner thicker skinnier faster slower quicker "
    "darker lighter brighter dimmer paler deeper bolder heavier softer rounder sharper squarer tighter looser closer "
    "further farther higher lower longer warmer cooler stronger weaker subtler louder quieter").split()
# verbs that carry their direction on their own
DIRECTION_VERBS = (r"increase|decrease|reduce|raise|shrink|enlarge|widen|darken|lighten|brighten|soften|thicken|tighten|"
                   r"loosen|halve|tone\s+(?:it\s+|this\s+|that\s+|them\s+)?down|turn\s+(?:it\s+|this\s+|that\s+|them\s+)?(?:up|down)|"
                   r"scale\s+(?:it\s+|this\s+|them\s+)?(?:up|down)|size\s+(?:it\s+|this\s+|them\s+)?(?:up|down)|"
                   r"crank\s+(?:it\s+|this\s+|that\s+)?up|dial\s+(?:it\s+|this\s+|that\s+)?(?:up|down|back)|"
                   r"bump\s+(?:it\s+|this\s+|that\s+|them\s+)?(?:up|down)|speed\s+(?:\w+\s+){0,3}?up|slow\s+(?:\w+\s+){0,3}?down")
# verb-or-adjective words: a direction only in the verb's place ("narrow the column", not "the narrow column")
AMBIGUOUS_VERBS = r"narrow|stretch|squeeze|grow|expand|dim|thin|cut|trim|double|boost|bump|lower"
VERB_OBJECT = r"(?:the|it|this|that|them|these|those|my|our|everything|all|up|down|out|back|a\s+bit|a\s+little|a\s+touch)\b"
VERB_LEAD = r"(?:^|please\s+|pls\s+|can\s+you\s+|could\s+you\s+|can\s+we\s+|let'?s\s+|just\s+|and\s+|also\s+|then\s+|to\s+|[,.;!]\s*)"
# "too <word>" / "not <word> enough": a direction only when the word is a size, amount, spacing, speed or strength
# (r7 review: "too plain", "too boxy", "too wordy" name no way to go — a complaint, so vague)
TOO_DIRECTIONAL = frozenset(
    "big large huge massive tiny small little short long tall wide narrow thin thick skinny fat many much few close tight "
    "far spread loose cramped crowded squished squashed squeezed high low dark light pale bright white black faint washed "
    "heavy harsh subtle loud quiet bold fast slow quick busy dim dull strong weak sharp intense visible hidden "
    "noticeable prominent".split())
# "more / less <word>": an adjective that names a style, not an amount, is no direction ("more modern")
MORE_STYLE = frozenset(
    "modern obvious useful fancy relaxing uplifting personalized personalised natural handwritten exciting professional "
    "fun interesting appealing premium elegant polished inviting friendly welcoming clean cohesive consistent playful "
    "stylish trendy fresh lively upscale luxurious attractive eye-catching catchy engaging inspiring creative unique "
    "original fitting relevant on-brand branded organic earthy warm cozy cosy calm calming vibrant fun-looking "
    "cheerful pretty beautiful nice nicer appropriate trustworthy legit legitimate high-end classy chic minimal "
    "minimalist simple".split())
NOT_DIRECTION_MORE = (r"(?<!read\s)(?<!learn\s)(?<!see\s)(?<!view\s)(?<!load\s)(?<!shop\s)(?<!out\s)(?<!find\s)(?<!no\s)"
                      r"(?<!any\s)")
ACTIONS = (r"hide|unhide|remove|delete|get\s+rid\s+of|dismissi?ble|closable|closeable|"
           r"let\s+(?:\w+\s+){1,2}?(?:close|dismiss|hide|skip)|(?:start|begin|load|open)s?\s+(?:with\s+)?(?:\w+\s+){0,3}?(?:closed|collapsed|expanded|open)\b|"
           r"collapsed|expanded|take\s+(?:it\s+|this\s+|that\s+|them\s+)?(?:out|off|away)|"
           r"(?:turn|switch)\s+(?:(?:it|this|that|them|the\s+[\w-]+(?:\s+[\w-]+)?)\s+)?(?:on|off)\b|"
           r"enable|disable|activate|deactivate|"
           r"reset|restore|revert|undo|clear\s+(?:the|it|out|this|that|my|our)|"
           r"cent(?:er|re)(?:ed)?\s+(?:the|it|this|that|them|everything|all|align)|cent(?:er|re)d|cent(?:er|ere|re)d|cent(?:er|re)-aligned|"
           r"left[- ]align(?:ed)?|right[- ]align(?:ed)?|align(?:ed)?\s+(?:it\s+|them\s+|everything\s+)?(?:to\s+the\s+)?(?:left|right|cent(?:er|re))|"
           r"uppercase|lowercase|all\s+caps|capitali[sz]e|bold|italic|italics|underline|underlined|"
           r"sticky|stay\s+at\s+the\s+top|reorder|duplicate|"
           r"move\s+(?:it\s+|this\s+|that\s+|them\s+|the\s+[\w-]+(?:\s+[\w-]+)?\s+)?(?:up|down|above|below|to\s+the\s+(?:top|bottom|left|right))|"
           r"swap(?!\s+out)\s+(?:\w+\s+){1,6}?(?:and|with)\b|open\s+in\s+a\s+new\s+tab|full[- ]width")
SHOW_VERB = r"(?:show|display)(?!\s+up)\b"
NOT_LEAD = r"(?:don'?t|do\s+not|stop|never|no\s+longer)\s+"
# "add a question about delivery times", "add a review from Sarah": an item with what it is about
ADD_ITEM = re.compile(r"\b(?:add|include|insert|put\s+in)\s+(?:a|an|another|one|more|new|\s)*\s*(?:question|faq|review|testimonial|"
                      r"quote|slide|card|item|link|tab|step|location|category|feature|perk)s?\s+(?:about|from|on|for|asking|saying|"
                      r"that\s+says|with|by|regarding)\b")
ADD_VERB = r"\b(?:add|include|put|insert)\b"
IMAGE_NOUNS = r"images?|photos?|pictures?|pics?|logo|icons?|favicon|banner\s+image|background\s+image|hero\s+image"
IMAGE_ATTRS = frozenset("color colour colors colours size sizes height width gap position layout alignment align opacity "
                        "border borders radius spacing padding margin text caption captions alt link links overlay ratio "
                        "fit shape style frame font title titles max placement columns rows".split())
# words that may stand between "change" and the picture it means ("change the big image at the top")
IMAGE_GAP_STOP = frozenset("how where what whether which why when gap spacing space between order direction size text "
                           "color colour width height link links tile button buttons in on of under above below with wide "
                           "tall fast slow under over to for from at by".split())
# a word after say / call / name … that is no value ("should say something else", "what it says on the bar")
NOT_A_TEXT = frozenset("something anything else different new better nicer well right properly more less easier "
                       "on in at the a an about here there it this that now so and but for to too when is if".split())
# a word after say / call / name … in reported speech ("people say they can't search"): no value
REPORTED_SPEAKERS = r"(?:people|customers|clients|shoppers|they|someone|somebody|everyone|everybody|folks|users|friends|i|we|you|he|she)"
UI_NOUNS = frozenset("faq seo google facebook instagram tiktok weedmaps leafly twitter x youtube linkedin shop menu home "
                     "footer header cta url i".split())
GENERIC_OPTIONS = frozenset("default none auto custom normal inherit true false yes no on off all other unset initial".split())
# a colour word that describes what is there now ("the grey text", "looks so white"): no value (r7 review)
COLOUR_DESCRIBES = r"(?:the|this|that|these|those|its|it's|our|my|their|looks?|looked|is|are|was|were|so|very|all|too|kinda|kind\s+of|pretty|really|such\s+a)"
SHAPE_VALUE = re.compile(r"\b(?:square|squared|round|rounded|pill|pill-shaped|circle|circular|serif|sans|sans-serif|"
                         r"monospace|monospaced)\b")
SORT_VALUE = re.compile(r"\bsort(?:ed)?\s+(?:\w+\s+){0,4}?by\b|\b(?:low(?:est)?|high(?:est)?|cheap(?:est)?)\s+(?:to\s+(?:high|low)(?:est)?|first)\b|"
                        r"\b(?:newest|oldest|latest)\s+first\b|\ba\s*-\s*z\b|\bz\s*-\s*a\b|\balphabetical(?:ly)?\b")
# a code or id the merchant types ("G-ABC123", "GTM-5XQ2"): letters and digits in one token
ID_TOKEN = re.compile(r"(?<![\w#/.@-])(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?![\w/.@-])")
# "link / point / go … to the products page": a page as a link's target is a value
PAGE_LINK = re.compile(r"\b(?:link|links|linked|linking|point|points|pointed|go|goes|send|sends|take|takes|lead|leads|open|opens)\s+"
                       r"(?:\w+\s+){0,3}?to\s+(?:the\s+|our\s+|my\s+)?[\w'&-]+(?:\s+[\w'&-]+)?\s+page\b")
COLOUR_LITERAL = re.compile(r"#[0-9a-f]{3,8}\b|\brgba?\(|\bhsla?\(", re.I)
NUMBER = re.compile(r"(?<![\w#./])\d+(?:\.\d+)?\s*(?:px|pt|em|rem|%|percent|pixels?|ms|s|seconds?|columns?|cols?|items?|"
                    r"products?|rows?|per\s+row|x)?(?![\w])", re.I)
INSTANCE_NUMBER = re.compile(r"\b(?:panels?|buttons?|blocks?|sections?|slides?|cards?|columns?|tabs?|steps?|items?|images?|"
                             r"photos?|pictures?|links?|titles?|headlines?|banners?|carousels?|content|location|store|option|"
                             r"number|no\.?|#)\s*#?\d+\b", re.I)
URLISH = re.compile(r"https?://|\bwww\.|\b[\w-]+\.(?:com|ca|net|org|io|co|shop|store|app|page|ly|me|us|uk)\b|"
                    r"[\w.+-]+@[\w-]+\.\w+|(?<![\w@])@[a-z0-9_][\w.]{1,}|\bmailto:|(?:^|\s)/[a-z0-9][\w/-]*", re.I)
QUOTED = re.compile(r"(?:^|(?<=[\s:(\[,]))(?:\"([^\"]{1,120})\"|'([^']{1,120})'|“([^”]{1,120})”|‘([^’]{1,120})’)"
                    r"(?=$|[\s.,!?;:)\]])")
TO_VALUE = re.compile(r"(?:\b(?:to|into|as|is|be|should\s+be|should\s+read|should\s+say)\s+|=\s*|:\s*)"
                      r"(\"|'|“|‘|[A-Z][\w&'’.-]*|[\w-]*\d[\w-]*)")
SAY_VALUE = re.compile(r"\b(?:says?|saying|reads?|reading|called|call\s+it|named|name\s+it|titled|labell?ed|rename(?:d)?\s+(?:\w+\s+){0,6}?to)"
                       r"\s+(?:it\s+)?([^\s,.!?;]+(?:\s+[^\s,.!?;]+){0,5})", re.I)
# "doesn't feel right", "right now": idioms whose option word is no value
IDIOMS = re.compile(r"\b(?:feels?|looks?|seems?|sits?|is|isn'?t|not|quite|all|just|reads?)\s+right\b|\bright\s+(?:now|away|here|there)\b|"
                    r"\b(?:at|on)\s+the\s+(?:top|bottom)\b|\bthe\s+(?:top|bottom)\s+of\b|\balright\b", re.I)


class Verdict(NamedTuple):
    vague: bool
    kind: str   # value | direction | action | none
    why: str


def norm(text: Any) -> str:
    """Lower case; apostrophes, quotes and whitespace normalised."""
    t = str(text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def _plain(v: Any) -> str:
    return norm(re.sub(r"<[^>]+>", " ", str(v if v is not None else ""))).strip(" '\".")


def _phrase(text: str, phrase: str) -> bool:
    p = _plain(phrase)
    if len(p) < 2:
        return False
    return re.search(r"(?<![\w-])" + re.escape(p) + r"(?![\w-])", text) is not None


# --- the three classes --------------------------------------------------------------------------------------------------

# say / call / name … words that NAME a thing the site shows ("the section called Our Story"); the others ("rename …
# to X", "should say X", "make it say X") ask for X
_DESCRIBING = frozenset("says saying reads reading called named titled labelled labeled".split())
_PRESCRIBING_LEAD = re.compile(r"(?:should|to|make\s+it|have\s+it|let\s+it|it\s+should|can\s+it|could\s+it|will|must)\s+$")
_CONNECTORS = frozenset("and but is are was looks look needs need should so it on in at for to with which that the a "
                        "an has have could would can".split())


def _names_thing(x: str, known: set[str]) -> bool:
    """`x` (the words after a describing word) is a name the site shows: the whole of it, or it followed by the rest of
    the sentence ("Our Story needs work"). Exact words only — "Home delivery" is no "Home" (r7 review)."""
    for k in known:
        if len(k) < 2:
            continue
        if x == k:
            return True
        if x.startswith(k + " ") and x[len(k) + 1:].split(" ", 1)[0] in _CONNECTORS:
            return True
    return False


def has_value(text: str, values: Iterable[str] = (), names: Iterable[str] = ()) -> str | None:
    raw = str(text or "").replace("’", "'").replace("‘", "'")
    t = norm(raw)
    known = {_plain(n) for n in names if _plain(n)}
    if COLOUR_LITERAL.search(raw):
        return "V1:colour_literal"
    for m in re.finditer(r"[a-z]+", t):
        w = m.group(0)
        if w not in COLOUR_WORDS:
            continue
        if re.search(r"\btoo\s+(?:\w+\s+)?" + w + r"\b", t):
            continue
        before = t[:m.start()].rstrip()
        if re.search(r"(?:^|\s)" + COLOUR_DESCRIBES + r"$", before) and not re.search(
                r"\b(?:to|into|be|in|use|make\s+(?:it|them|this|that)|go|paint|try)\s+(?:\w+\s+)?$", before):
            continue  # "the grey text … should change", "looks so white": what is there now
        return f"V1:colour:{w}"
    no_instance = INSTANCE_NUMBER.sub(" ", t)
    m = NUMBER.search(no_instance)
    if m:
        return f"V2:number:{m.group(0).strip()}"
    m = ID_TOKEN.search(INSTANCE_NUMBER.sub(" ", raw))
    if m and not re.fullmatch(r"\d+(?:st|nd|rd|th|px|pt|em|rem|ms|s|x|k)?|[a-z]\d|h\d", m.group(0), re.I):
        return f"V2:id:{m.group(0)[:30]}"
    for m in SHAPE_VALUE.finditer(t):
        if re.search(r"\b(?:how|the|this|that|these|those|its|so|too|very|kinda|pretty)\s+$", t[:m.start()]):
            continue  # "change how round the buttons are", "the round logo": what is there now
        return f"V6:shape:{m.group(0)}"
    m = SORT_VALUE.search(t)
    if m:
        return f"V6:sort:{m.group(0)[:30]}"
    for m in QUOTED.finditer(raw):
        q = _plain(next(g for g in m.groups() if g is not None))
        if q and q not in known:
            return f"V3:quote:{q[:40]}"
    m = URLISH.search(raw)
    if m:
        return f"V4:link:{m.group(0).strip()[:40]}"
    m = PAGE_LINK.search(t)
    if m:
        return f"V4:page:{m.group(0)[:40]}"
    for m in TO_VALUE.finditer(raw):
        tok = m.group(1)
        if tok in ("\"", "'", "“", "‘"):
            return "V5:to_quote"
        w = tok.lower().strip(".'")
        if w in UI_NOUNS or w in ("i", "i'm", "i'd"):
            continue
        if re.match(r"\s*is\b", m.group(0), re.I):  # "the bit that is Shop now": a describing "is" may name a thing
            rest = _plain(re.split(r"[,.!?;]", raw[m.start(1):m.start(1) + 80])[0])
            if _names_thing(rest, known):
                continue
        return f"V5:to:{tok[:30]}"
    for m in SAY_VALUE.finditer(t):
        x = m.group(1).strip()
        first = x.split()[0] if x else ""
        if not first or first in NOT_A_TEXT:
            continue
        before = t[:m.start()]
        if re.search(r"\b" + REPORTED_SPEAKERS + r"\s+(?:all\s+|always\s+|keep\s+|kept\s+)?$", before):
            continue  # "people say they can't search": what someone said, no text to set
        word = m.group(0).split()[0]
        if word in _DESCRIBING and not _PRESCRIBING_LEAD.search(before) and _names_thing(x, known):
            continue  # "the section called Our Story": a name the site shows
        return f"V5:text:{x[:30]}"
    t6 = IDIOMS.sub(" ", t)
    for v in values:
        p = _plain(v)
        if not p or p in GENERIC_OPTIONS or len(p) < 3:
            continue
        if _phrase(t6, p):
            return f"V6:option:{p[:30]}"
    return None


def _too_direction(t: str) -> str | None:
    for m in re.finditer(r"\b(?:way\s+|much\s+|far\s+|a\s+bit\s+|a\s+little\s+)?too\s+([a-z][\w-]*)", t):
        if m.group(1) in TOO_DIRECTIONAL:
            return f"D3:too:{m.group(1)}"
    for m in re.finditer(r"\b(?:not|isn'?t|aren'?t|wasn'?t)\s+((?:\w+\s+){0,2}?)enough\b", t):
        words = m.group(1).split()
        if words and words[-1] in TOO_DIRECTIONAL | {"contrast", "space", "room", "padding"}:
            return "D3:not_enough"
    if re.search(r"\bnot\s+enough\s+(?:space|room|padding|spacing|contrast|gap|margin|breathing)\b", t):
        return "D3:not_enough"
    return None


def has_direction(text: str) -> str | None:
    t = norm(text)
    t = re.sub(r"\bno\s+longer\b|\b(?:further|farther)\s+(?:down|up|along|below|above|on)\b|\bdouble[- ]check\b", " ", t)
    d = _too_direction(t)
    if d:
        return d
    m = re.search(r"\b(" + "|".join(COMPARATIVES) + r")\b", t)
    if m:
        return f"D1:{m.group(1)}"
    for m in re.finditer(NOT_DIRECTION_MORE + r"\b(more|less|fewer)\b(?!\s+(?:about|info|information|details?|options|"
                         r"than\s+that|people|customers|sales|traffic|visitors|orders|trust|engagement)\b)", t):
        nxt = re.match(r"\s+([a-z][\w-]*)", t[m.end():])
        if nxt and nxt.group(1) in MORE_STYLE:
            continue  # "more modern": a style, no amount
        return f"D2:{m.group(1)}"
    m = re.search(r"\b(" + DIRECTION_VERBS + r")\b", t)
    if m:
        return f"D4:{re.sub(r'\s+', ' ', m.group(1))}"
    m = re.search(VERB_LEAD + r"(" + AMBIGUOUS_VERBS + r")\s+" + VERB_OBJECT, t)
    if m:
        return f"D4:{m.group(1)}"
    return None


def has_action(text: str, *, bool_labels: Iterable[str] = (), section_names: Iterable[str] = ()) -> str | None:
    t = norm(text)
    m = re.search(r"\b(" + ACTIONS + r")\b", t)
    if m:
        return "action:" + re.sub(r"\s+", " ", m.group(1))[:30]
    m = re.search(VERB_LEAD + SHOW_VERB, t) or re.search(r"\b" + NOT_LEAD + SHOW_VERB, t)
    if m:
        return "action:show"
    m = ADD_ITEM.search(t)
    if m:
        return "action:add_item"
    if image_request(t):
        return "action:pick_asset"
    if re.search(ADD_VERB, t):
        for s in section_names:
            if _phrase(t, s):
                return f"action:add:{_plain(s)[:30]}"
        for lab in bool_labels:
            core = re.sub(r"^(?:show|hide|enable|disable|display|use|include)\s+", "", _plain(lab))
            if len(core) >= 3 and _phrase(t, core):
                return f"action:add:{core[:30]}"
    return None


def image_request(text: str) -> bool:
    """An image asked to be changed, replaced, uploaded or swapped: the picture is the verb's object ("change the
    big image at the top", "a new logo", "upload a banner photo"), not a place ("the gap on the banner image")."""
    t = norm(text)
    for m in re.finditer(r"\b(" + IMAGE_NOUNS + r")\b(?:\s+([a-z]+))?", t):
        if m.group(2) in IMAGE_ATTRS:
            continue
        before = t[:m.start()].split()
        if "upload" in before:
            return True
        for i in range(len(before) - 1, max(-1, len(before) - 6), -1):
            w = before[i]
            if w in ("change", "replace", "update", "swap", "switch", "new", "different", "another", "add", "insert"):
                return True
            if w in IMAGE_GAP_STOP:
                break
        after = t[m.end():]
        if re.match(r"\s+(?:needs?\s+to\s+be|should\s+be)\s+(?:changed|replaced|swapped|updated)\b", after):
            return True
    return False


def classify(text: Any, *, values: Iterable[str] = (), bool_labels: Iterable[str] = (),
             section_names: Iterable[str] = (), names: Iterable[str] = ()) -> Verdict:
    """VAGUE iff no VALUE, no DIRECTION and no ACTION (module docstring). `values`: the case's option and oracle
    vocabulary (values_for); `bool_labels`: its toggles' labels and `section_names`: the section types a request may
    add; `names`: texts the site already shows (a quote of one names the thing, it gives no value)."""
    s = str(text or "")
    why = has_value(s, tuple(values), tuple(names))
    if why:
        return Verdict(False, "value", why)
    why = has_direction(s)
    if why:
        return Verdict(False, "direction", why)
    why = has_action(s, bool_labels=bool_labels, section_names=section_names)
    if why:
        return Verdict(False, "action", why)
    return Verdict(True, "none", "vague")


# --- direction by field type -----------------------------------------------------------------------------------------------

_UP = ("bigger larger taller wider thicker longer higher heavier bolder louder stronger further farther looser more "
       "increase raise enlarge widen thicken loosen grow expand stretch double boost").split()
_DOWN = ("smaller tinier shorter narrower thinner lower lighter-weight closer tighter softer quieter weaker subtler less "
         "fewer decrease reduce shrink tighten halve narrow squeeze cut trim thin").split()
_DARKER = "darker deeper dimmer richer darken dim".split()
_LIGHTER = "lighter brighter paler lighten brighten".split()
_TOO_UP = "small tiny short narrow thin skinny few little close tight cramped crowded squished squashed low".split()
_TOO_DOWN = "big large tall wide thick fat long many much spread far loose high busy".split()
_TOO_DARKER = "light pale bright white faint washed".split()
_TOO_LIGHTER = "dark black heavy harsh".split()
_ENOUGH_UP = "big large wide tall long bold thick visible".split()
_ENOUGH_DARKER = "dark bold strong".split()
_ENOUGH_LIGHTER = "light bright".split()


# an opacity / transparency field: "fade it more" is a LOWER number, "more" says up — no guess (r7 review: r6's
# "fade the header background image a lot more" expects down)
OPACITY_KEY = re.compile(r"opacity|transparen|alpha|fade", re.I)
OPACITY_WORDS = re.compile(r"\b(?:fade[ds]?|fading|faded|transparent|transparency|see[- ]through|opacity|opaque|"
                           r"translucent)\b")


def direction_of(text: Any, field_type: str, key: str | None = None) -> str | None:
    """The direction a turn asks for on a field of `field_type`: up / down (number, range), darker / lighter (colour);
    None when the type has no direction, the words point both ways or nowhere clear ("too subtle"), or the field
    (`key`) or the words are about opacity (the number runs against the words)."""
    t = norm(text)
    colour = field_type == "color"
    number = field_type in ("range", "number")
    if not (colour or number):
        return None
    if number and (OPACITY_KEY.search(str(key or "")) or OPACITY_WORDS.search(t)):
        return None
    found: set[str] = set()
    for m in re.finditer(r"\b(?:way\s+)?too\s+([a-z]+)", t):
        w = m.group(1)
        if colour and w in _TOO_DARKER:
            found.add("darker")
        elif colour and w in _TOO_LIGHTER:
            found.add("lighter")
        elif number and w in _TOO_UP:
            found.add("up")
        elif number and w in _TOO_DOWN:
            found.add("down")
    for m in re.finditer(r"\bnot\s+(?:\w+\s+)?([a-z]+)\s+enough\b", t):
        w = m.group(1)
        if number and w in _ENOUGH_UP:
            found.add("up")
        elif colour and w in _ENOUGH_DARKER:
            found.add("darker")
        elif colour and w in _ENOUGH_LIGHTER:
            found.add("lighter")
    t2 = re.sub(r"\b(?:way\s+)?too\s+[a-z]+|\bnot\s+(?:\w+\s+)?[a-z]+\s+enough\b", " ", t)
    words = re.findall(r"[a-z]+", t2)
    phrases = set(words) | {" ".join(p) for p in zip(words, words[1:])}
    if colour:
        if phrases & set(_DARKER):
            found.add("darker")
        if phrases & set(_LIGHTER):
            found.add("lighter")
    if number:
        if phrases & set(_UP) or re.search(r"\b(?:bump|turn|scale|size|crank|dial)\s+(?:\w+\s+){0,4}?up\b", t2):
            found.add("up")
        if phrases & set(_DOWN) or re.search(r"\b(?:bump|turn|scale|size|dial|tone)\s+(?:\w+\s+){0,4}?(?:down|back)\b", t2):
            found.add("down")
        if "bump" in phrases and not re.search(r"\bbump\s+(?:\w+\s+){0,4}?(?:up|down)\b", t2) and "down" not in found:
            found.add("up")
    return next(iter(found)) if len(found) == 1 else None


# --- evidence for a write (r7 review: a turn the rule calls clear is relabelled as a write only on evidence) ----------

_TOGGLE_STOP = frozenset("show hide enable disable display use include allow the and for with".split())


def oracle_literals(ops: Iterable[dict] | None) -> list[str]:
    """Every literal the ops write — strings (lower case, tags stripped; hex colours kept), numbers; no booleans."""
    out: list[str] = []

    def walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            p = _plain(v)
            if p and len(p) >= 2:
                out.append(p)
        elif isinstance(v, (int, float)):
            out.append(f"{v:g}" if isinstance(v, float) else str(v))
    for o in ops or []:
        if isinstance(o, dict):
            walk(o.get("values"))
            walk(o.get("item"))
            walk(o.get("value"))
    return out


def _literal_in(t: str, lit: str) -> bool:
    if re.fullmatch(r"-?\d+(?:\.\d+)?", lit):
        return re.search(r"(?<![\d.])" + re.escape(lit) + r"(?![\d])", t) is not None
    return re.search(r"(?<![a-z0-9])" + re.escape(lit) + r"(?![a-z0-9])", t) is not None


def write_evidence(text: Any, field=None, *, oracle_ops: Iterable[dict] | None = None,
                   verdict: Verdict | None = None) -> str | None:
    """Why a CLEAR turn (classify: not vague) may become a write case on `field` — or None, and it may not (it stays
    out: an ask label would break the rule, an arbitrary write teaches the wrong thing — "the headings look too plain"
    became headingFont in one r6 case and buttonFont in another):

      oracle     a literal the source's oracle writes is in the turn ("…should read Local and proud", "24px")
      direction  the field is a colour / number and direction_of reads one way for it
      option     a select field: one of its own options is in the turn, or a sort order on a sort field
      toggle     an action on a bool field whose label shares a word with the turn ("remove the cart buttons" on
                 "Show add to cart"; never "show what payments we accept" on "E-transfer")
      literal    no oracle (a pool ask): a colour literal, a number, a quote or a link (V1–V4)"""
    t = norm(text)
    if oracle_ops:
        for lit in oracle_literals(oracle_ops):
            if _literal_in(t, lit):
                return f"oracle:{lit[:30]}"
    ft = getattr(field, "type", None)
    if ft in ("color", "range", "number"):
        d = direction_of(text, ft, getattr(field, "key", None))
        if d:
            return f"direction:{d}"
    v = verdict or classify(text)
    if ft in ("select", "enum"):
        t6 = IDIOMS.sub(" ", t)
        for o in list(getattr(field, "enum", None) or []) + list((getattr(field, "enum_labels", None) or {}).values()):
            p = _plain(o)
            if p and len(p) >= 3 and p not in GENERIC_OPTIONS and _phrase(t6, p):
                return f"option:{p[:30]}"
        if v.why.startswith("V6:sort") and re.search(r"sort|order", str(getattr(field, "key", "")), re.I):
            return "option:sort"  # "sort the carousel by THC, highest first" on sortBy (thc-desc)
    if ft == "bool" and v.kind == "action":
        words = {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", _plain(getattr(field, "label", "") or ""))} - _TOGGLE_STOP
        mine = {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", t)}
        if words & mine:
            return "toggle:" + sorted(words & mine)[0]
    if not oracle_ops and v.kind == "value" and re.match(r"V1:colour_literal|V2:number|V3:|V4:", v.why):
        return "literal:" + v.why
    return None


# --- a case's vocabulary ------------------------------------------------------------------------------------------------------

_FONT_KEY = re.compile(r"font(?:family)?$", re.I)


def _case_fields(case: dict, m) -> list:
    """The manifest Fields a case names: its ops' and expected keys' writes, else its target + fields."""
    from .exam import touches_of
    from .retrain.rebase import field_of
    out, seen = [], set()

    def add(f):
        if f is not None and id(f) not in seen:
            seen.add(id(f))
            out.append(f)
    e = case.get("expected") or {}
    meta = case.get("meta") or {}
    for o in list(e.get("ops") or []) + list(meta.get("oracle_ops") or []):
        if isinstance(o, dict):
            for t in touches_of(o):
                if t.get("key") and not str(t["key"]).startswith("__"):
                    add(field_of(m, t.get("target"), str(t.get("list") or ""), str(t["key"]).split(".")[0]
                                 if t.get("list") else str(t["key"])))
    for k in e.get("keys") or []:
        if isinstance(k, dict) and k.get("key"):
            add(field_of(m, k.get("target"), str(k.get("list") or ""), str(k["key"])))
    tgt = case.get("target") if isinstance(case.get("target"), dict) else None
    for key in case.get("fields") or []:
        key = str(key)
        f = field_of(m, tgt, "", key) if tgt else None
        if f is None:
            f = m.global_fields().get(key)
        if f is None:
            for _, ff in m.iter_fields():
                if ff.key == key:
                    f = ff
                    break
        add(f)
    return out


def font_names(m) -> list[str]:
    """Every font family the manifest's font fields offer (h006 "Inter" is on the blog's font list, not the site's)."""
    out: set[str] = set()
    for _, f in m.iter_fields():
        if f.type in ("select", "enum") and _FONT_KEY.search(f.key):
            out |= {str(v) for v in f.enum if v} | {str(v) for v in f.enum_labels.values() if v}
    return sorted(out)


def _asset_names(case: dict) -> list[str]:
    from .state import load_assets_fixture
    try:
        assets = load_assets_fixture()
    except Exception:  # noqa: BLE001 - no fixture: no asset vocabulary
        assets = []
    out = []
    for a in assets + list(case.get("assets") or []):
        if not isinstance(a, dict):
            continue
        stem = re.sub(r"\.\w+$", "", str(a.get("public_url") or "").rsplit("/", 1)[-1])
        stem = re.sub(r"[-_]\d+(?:x\d+)?$", "", stem)
        if stem:
            out.append(stem.replace("-", " ").replace("_", " "))
        if a.get("alt"):
            out.append(str(a["alt"]))
    return out


def _ref_names(case: dict, kind: str) -> list[str]:
    from .state import load_context_fixture
    lists: list = []
    try:
        lists.append((load_context_fixture() or {}).get(kind))
    except Exception:  # noqa: BLE001
        pass
    for src in (case.get("context") or {}, case.get("context_extras") or {}, case.get("option_lists") or {}):
        if isinstance(src, dict):
            lists.append(src.get(kind))
    out = []
    for lst in lists:
        for x in lst or []:
            if isinstance(x, dict) and x.get("name"):
                out.append(str(x["name"]))
    return out


def _page_names(case: dict) -> list[str]:
    st = case.get("site_state") or {}
    out = []
    for p in st.get("customPages") or []:
        if isinstance(p, dict):
            out += [str(p.get("title") or ""), str(p.get("slug") or "")]
    for slug, v in (st.get("pageVisibility") or {}).items():
        if isinstance(v, dict):
            out += [str(v.get("headerNavName") or ""), str(v.get("footerNavName") or "")]
    return [x for x in out if x]


def _oracle_literals(ops: Iterable[dict]) -> list[str]:
    from .exam import touches_of  # noqa: F401 - the ops' shape
    out = []

    def walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            p = _plain(v)
            if p and len(p) >= 2 and not p.startswith("#"):
                out.append(p)
        elif isinstance(v, (int, float)):
            out.append(str(v))
    for o in ops or []:
        if isinstance(o, dict):
            walk(o.get("values"))
            walk(o.get("item"))
    return out


def values_for(case: dict, m, *, oracle_ops: Iterable[dict] | None = None) -> tuple[str, ...]:
    """V6 + V7: the options of the case's fields (enum values and labels; the manifest's fonts for a font field; asset
    names for an image field; the store's categories / tags / brands / locations for a reference field; page names for
    a link / page field) and the literals `oracle_ops` write."""
    out: list[str] = []
    for f in _case_fields(case, m):
        out += [str(v) for v in f.enum if v not in (None, "")] + [str(v) for v in f.enum_labels.values() if v]
        if _FONT_KEY.search(f.key) and f.type in ("select", "enum"):
            out += font_names(m)
        if f.type == "image":
            out += _asset_names(case)
        ref = str(f.ref or "")
        if ref in ("category", "tag", "brand", "location") or re.search(r"(?:category|tag|brand)(?:Slug|Id)?$", f.key):
            kind = {"category": "categories", "tag": "tags", "brand": "brands", "location": "locations"}.get(
                ref or re.search(r"(category|tag|brand)", f.key, re.I).group(1).lower())
            out += _ref_names(case, kind)
        if re.search(r"(?:link|url|href)", f.key, re.I) and f.type in ("text", "select", "enum"):
            # a page as a link's target: "point it to the contact page" (never the bare word: "not sure about …")
            out += [f"{pre} {p}" for p in _page_names(case) for pre in ("to", "to the", "to our", "to my", "at")]
    if oracle_ops:
        out += _oracle_literals(oracle_ops)
    return tuple(dict.fromkeys(v for v in out if v))


def bool_labels_for(case: dict, m) -> tuple[str, ...]:
    """The labels of the toggles in the containers the case names ("add the search bar": the bool "Show search")."""
    from .retrain.rebase import container_of
    out: list[str] = []
    tgts = [case.get("target")] + [o.get("target") for o in ((case.get("expected") or {}).get("ops") or [])
                                   if isinstance(o, dict)]
    seen = set()
    for t in tgts:
        if not isinstance(t, dict):
            continue
        _, c = container_of(m, t)
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        out += [str(f.label) for f in c.fields.values() if f.type == "bool" and f.label]
    return tuple(out)


SECTION_SYNONYMS = {"productCarousel": ("product carousel", "carousel"), "newsletter": ("newsletter", "sign-up", "signup"),
                    "testimonials": ("testimonials", "reviews section"), "gallery": ("gallery", "photo gallery"),
                    "faq": ("faq section",), "contentBlock": ("content block",), "heroBanner": ("hero banner",),
                    "instagramFeed": ("instagram feed",), "blogSection": ("blog section",)}


def section_names_for(m) -> tuple[str, ...]:
    out: list[str] = []
    types = m.section_types() if callable(m.section_types) else m.section_types
    for t in types:
        out.append(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t).lower())
        out += list(SECTION_SYNONYMS.get(t, ()))
    return tuple(dict.fromkeys(out))


def names_for(case: dict, limit: int = 80) -> tuple[str, ...]:
    """Texts the case's site already shows (its state's short strings): quoting one names a thing."""
    out: list[str] = []

    def walk(v, depth=0):
        if depth > 8:
            return
        if isinstance(v, dict):
            for x in v.values():
                walk(x, depth + 1)
        elif isinstance(v, list):
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, str):
            p = _plain(v)
            if 2 <= len(p) <= limit and not p.startswith(("http", "#", "/", "{")):
                out.append(p)
    st = case.get("site_state") or {}
    walk(st)
    try:  # the nav's built-in names ("the 'Shop' item in the header") are no stored string
        from .state import BUILTIN_PAGE_NAMES, nav_summaries
        out += [_plain(r.get("label")) for r in nav_summaries(st.get("pageVisibility"), st.get("customPages"))]
        out += [_plain(n) for v in BUILTIN_PAGE_NAMES.values() for n in (v if isinstance(v, (list, tuple)) else [v])]
    except Exception:  # noqa: BLE001 - a state the summaries cannot read: its strings only
        pass
    return tuple(dict.fromkeys(x for x in out if x))


def classify_case(case: dict, m, *, text: str | None = None, oracle_ops: Iterable[dict] | None = None) -> Verdict:
    """classify() with the case's whole vocabulary (values_for, its toggles, the manifest's section types and the texts
    its site shows). `text`: the turn to judge (default the case's request)."""
    t = case.get("request") if text is None else text
    return classify(t, values=values_for(case, m, oracle_ops=oracle_ops), bool_labels=bool_labels_for(case, m),
                    section_names=section_names_for(m), names=names_for(case))


def field_type_of(case: dict, m) -> str | None:
    """The type of the one field a case names, or None (no field, or several of different types)."""
    types = {f.type for f in _case_fields(case, m)}
    return next(iter(types)) if len(types) == 1 else None


ASK_FAMILIES = ("complaint", "text_missing", "choice_missing", "which_one", "page_title_missing", "rest")


def ask_family_of(case: dict, m=None) -> str:
    """The ask family a case belongs to (r7 spec §3's floors): meta.ask_family when set; a clarify / which-one ask;
    a no-title create_page; else by the named field's type (text → text_missing, a colour / option / font →
    choice_missing); `rest` otherwise."""
    meta = case.get("meta") or {}
    if meta.get("ask_family"):
        return str(meta["ask_family"])
    if (case.get("score") or {}).get("mode") == "clarify" or meta.get("ask_reason") == "which_one":
        return "which_one"
    if meta.get("capability") == "create_page":
        return "page_title_missing"
    if m is not None:
        ft = field_type_of(case, m)
        if ft in ("text", "richtext"):
            return "text_missing"
        if ft in ("color", "select", "enum"):
            return "choice_missing"
    return "rest"
