"""contrast — family F8: contrast cases for the look-alike fields both r4 and r5 confuse (r5why PLAN §2b).

Each PAIR is two or more SIDES — fields a merchant's words can point at
interchangeably ("the link colour" in Typography vs the Shop Menu's "Link
Color"; the site's body font vs one section's body copy font …). A side is
asked several ways, each naming what makes it THAT field:

  literal   the value is typed (hex, px, a font, a quoted name) → scored exact
  relative  "a bit darker / bigger" → scored by keys + direction
  contrast  the request names the other side to rule it out ("just the shop menu, not the site's links")
  focus     the side's panel or section is open and the request names only the field's label

The words come from the field metadata, never from the exam: every template
fills `{label}` from the manifest label or shares a content word with the
field's label / help / container (`check_templates` enforces it), the values
come from the field's type, options and range, and every case goes through
the same validation as the authored families:

  oracle    r4cases.remeasure (humanized state, contexts derived, exact diff measured); a write whose gate is
            CLOSED in the case's state (r4score.closed_gate_writes on the engine's `hidden` detail) drops the case —
            r5's F7 taught 22 hidden writes; a gated field whose gate is open (the state seeds it) is kept
  mode      r4cases.authored_values → to_keys when a literal value is not one the merchant typed
  holdout   a case that writes a generator hold-out key (manifest.is_holdout on r4cases.write_ids) is dropped:
            TypographyColorsSection.navFontSize and the about / blog header names ARE the F0H exam's unseen keys
  leak      leak.py rules 1–3 against the exam sets and the study's real conversations, plus a margin: a text
            within Jaccard NEAR_EXAM (0.5) of any exam text is dropped too, plus the exam-ANSWER rule
            (exam_answer_clash): a case that writes the same field as an exam item, or a look-alike of it (any side
            of the same pair), with the same value the exam item states, or — for a per-page key
            (pageVisibility.<page>.…, orderTypeText.<page>.…) — for the same page, is dropped. The exam's wording
            can be far away (h019 "rename Shop in the navigation to Menu" vs "change Shop to 'Menu' in the footer
            only": Jaccard 0.33) while the case still teaches the exam's answer
  dedup     r4cases.dedup_key against the pool and the family itself

Two more groups ride along (r6-3, the user's 2026-10-15 decisions), in id spaces of their own (contrast1-F8-08nn,
-09nn) so no group's ids move when another grows:

  site_fonts  "the font" with a value and no part of the site named ("switch all the fonts to {v}"): the answer
              writes EVERY site font (r4score.to_all_fonts on the case's state; scored by r4score.score_site_fonts)
  vague       a field's label with no value, no direction and nothing to act on ("the {label} doesn't feel right"):
              mode `ask` — the right answer is a question, any write is a guess (r4score.score_ask). One per field
              of the pairs above (its metadata words: the manifest label); a field the generator holds out is skipped

Two capability groups (r6-4, the create_page tool and the dropdown option lists of 2026-10-15; "tools first": they
train in r6), ids from PAGE_BASE (contrast1-F8-10nn) and LIST_BASE (-11nn), built from PAGE_JOBS / LIST_JOBS:

  create_page   mode `page` (exam.score_page, on the op): create (title only; with a canonical slug; draft;
                in_nav), create plus sections in the same reply (the section adds on the new page are its own
                writes), collisions with a page the site has (a custom page in the context, a built-in page —
                with and without a same-reply section add, which the batch stop must block), reserved and
                non-canonical slugs, a template that can't show custom pages (declines: nothing written, the reply
                says why) — plus rename / delete / duplicate still report_unsupported where=pages (mode where),
                "add a section, not a page" (mode keys: a create_page there is a stray write) and "our LinkedIn
                page <url>" (mode keys: a social link, not a site page). The oracle measures
                every case on the round's engine: a create must apply, a decline's attempt must be refused with the
                expected code (E_SLUG / E_TARGET) — an engine without create_page drops the group
  list_options  mode `list` (r4score.score_listing): "what tags / brands / categories can I use for this
                carousel" on a focused product carousel, the case's lists in `option_lists` (tags, brands: the
                host's lists, also the engine ctx's) or `context_extras` (categories) — short (named in
                full), over 15 (15 names then "...and N more"), cut at the context's 100 (+ the dropdown), empty
                (say so), unknown (never say none; point at the dropdown); asked on the carousel or plainly
                ("which brands do we carry?", focus "plain"). The case is dropped when the derived context does not
                carry the list as the case means it (state.context_of's mirror)

Each capability case is its own split group (meta.group "<group>:<case id>"; meta.cap_group the report's group):
r4dataset hashes the group, and one name per behaviour sent a whole behaviour to validation or to train.
capability_report_cases builds eval's report-only `capability:built` set from CAPH_PAGE_JOBS / CAPH_LIST_JOBS.

and the follow-up family's vague FIRST turns (a pool case's history[0] the author marked `ask` / `offer`) become
single-turn `ask` cases of their own (ask_first_cases: ids `askfirst1-<family>-<source id>`, the source's state,
page, view and split group). Collect masks a follow-up's earlier replies out of the loss, so before these the
only trained "ask instead of guessing" turns were the ~7 clarify rows; the r5 drift went the other way.

Deterministic: the same manifest gives the same cases, no model call. Ids (`contrast1-F8-<nnnn>`, the
generator's version and the case's place in SIDES) do not depend on the manifest, so carry-forward can carry
a case whose request, state and answer did not change from one round to the next (a changed one is re-asked).

    python -m souschef_lab.contrast --manifest <manifest.json> --out f8.jsonl [--why <study dir>] \\
        [--pool <cases.jsonl>] [--report f8-report.json]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

FAMILY = "F8"
GENERATOR = "contrast-1"
TEMPLATES = ("vintage", "vintage", "bold", "boutique", "cozy", "editorial", "elegant", "playful", "pluto", "retro", "sweet")
PANEL_SECTION = {"BrandingSection": "branding", "HeaderSection": "headerMenu", "ShopMenuSection": "shopMenu",
                 "FooterSection": "footerMenu", "TypographyColorsSection": "typography", "ButtonsSection": "buttons",
                 "ProductCardSection": "productCard", "SocialMediaSection": "social", "LayoutSection": "layout"}
PALETTE = ("#1f6f43", "#b8325a", "#2b4c7e", "#d97706", "#4b5563", "#7c3aed", "#0e7490", "#be123c", "#365314",
           "#92400e", "#1e3a8a", "#6d28d9", "#0f766e", "#9f1239", "#3f6212", "#a16207")
FONT_SKIP = ("blambot-fx-pro",)
STOP = frozenset("a an the to of in on for and or my our your it this that be is are i we me please just only "
                 "not keep leave make set change can could should would use as at with so".split())


def h8(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


@dataclass(frozen=True)
class Side:
    """One field of a pair. `templates`: (style, text) with {v} (the value), {label} (the manifest label, lower
    case), {name} (a nav item's display name, from the manifest label) and {url}."""
    pair: str
    name: str
    kind: str                     # global | page | branding | section | list
    container: str                # panel / page slug / section id / "" for branding
    key: str                      # a field key; "{page}" is filled from `pages`; a list key for kind "list"
    templates: tuple
    n: int
    page: str = "home"            # the page the merchant is on
    pages: tuple = ()             # the nav pages a pageVisibility key is asked for
    seed: tuple = ()              # ((panel, key, value), …) written into the state first (a gate the field needs)
    values: tuple = ()            # a value bank for text fields, or ((page, bank), …) per `pages` member
    richtext: bool = False
    words: str = ""               # extra metadata words a template may use (the container's own name)


SHOP_MODERN = (("ShopMenuSection", "shopMenuStyle", "modern"),)
# value banks keyed by the page / option a pageVisibility key is asked for: a name that fits THAT link
# (no "Menu" for the Shop page: it is exam item h019's answer — exam_answer_clash drops any such case anyway)
NAV_NAMES = (("products", ("Shop All", "Browse", "Our Goods")), ("contact", ("Get in Touch", "Say Hello", "Reach Us")),
             ("faq", ("Help", "Questions", "Need Help?")), ("locations", ("Find Us", "Our Stores", "Visit")),
             ("loyalty", ("Rewards", "Perks", "Points Club")), ("about", ("Our Story", "Who We Are")))
CHECKOUT_TITLES = (("local_delivery", ("Doorstep Delivery", "Same-Day Delivery", "Delivered to You")),
                   ("local_pickup", ("Grab It In Store", "Express Pickup", "In-Store Pickup")),
                   ("mail_order", ("Ship It To Me", "Canada-Wide Shipping", "Mail Delivery")))
NEAR_EXAM = 0.5  # a margin under leak.py's rule 1 (0.6): a generated text this close to an exam text is dropped

SIDES: tuple[Side, ...] = (
    # 1. links: the site's text links vs the shop menu's links (both labelled "Link Color"), and the navigation
    #    font vs the shop menu font
    Side("links", "site_link_color", "global", "TypographyColorsSection", "linkColor", (
        ("literal", "change the {label} for the text links across the site to {v}"),
        ("literal", "links inside my page text should be {v}"),
        ("literal", "typography {label}: {v}"),
        ("relative", "make the text links on the site a little darker"),
        ("relative", "the links in my paragraphs could be a bit lighter"),
        ("contrast", "set the {label} for links in the page text to {v}, leave the shop menu alone"),
        ("focus", "{label} {v} please"),
        ("focus", "set the {label} here to {v}"),
    ), n=8, seed=SHOP_MODERN, words="typography colors text site"),
    Side("links", "shop_menu_link_color", "global", "ShopMenuSection", "shopMenuTextColor", (
        ("literal", "make the shop menu {label} {v}"),
        ("literal", "category links in the shop menu should be {v}"),
        ("literal", "shop menu links to {v}"),
        ("relative", "shop menu links a touch darker please"),
        ("contrast", "change the {label} to {v} in the shop menu only, not the links on the rest of the site"),
        ("focus", "change the {label} here to {v}"),
        ("focus", "{label}: {v}"),
    ), n=8, seed=SHOP_MODERN, words="shop menu category"),
    Side("links", "site_nav_font", "global", "TypographyColorsSection", "navFont", (
        ("literal", "use {v} for the {label} in the header and footer"),
        ("literal", "set the font of the {label} to {v}"),
        ("contrast", "{v} for the main {label}, not the shop menu"),
        ("focus", "{label} font: {v}"),
        ("literal", "switch the {label} font to {v}"),
    ), n=7, seed=SHOP_MODERN, words="typography header footer navigation"),
    Side("links", "shop_menu_font", "global", "ShopMenuSection", "shopMenuFont", (
        ("literal", "change the shop menu {label} to {v}"),
        ("literal", "use {v} in the shop menu"),
        ("contrast", "shop menu links in {v}, keep the header navigation font as it is"),
        ("focus", "set the {label} to {v}"),
    ), n=7, seed=SHOP_MODERN, words="shop menu"),
    Side("links", "shop_menu_link_size", "global", "ShopMenuSection", "shopMenuFontSize", (
        ("literal", "shop menu {label} {v}px"),
        ("literal", "make the shop menu links {v}px"),
        ("relative", "the links in the shop menu are a bit small, bump them up"),
        ("relative", "shop menu link text slightly smaller"),
        ("contrast", "set the shop menu {label} to {v}px but leave the header navigation links alone"),
        ("focus", "{label} {v}"),
    ), n=7, seed=SHOP_MODERN, words="shop menu link"),
    # 2. fonts: the whole site's body font vs one section's body copy font
    Side("fonts", "site_body_font", "global", "TypographyColorsSection", "bodyFont", (
        ("literal", "switch the paragraph font across the whole site to {v}"),
        ("literal", "use {v} for body text everywhere"),
        ("literal", "site-wide body font: {v}"),
        ("contrast", "change the body font for the whole site to {v}, not just one section"),
        ("focus", "set {label} to {v}"),
    ), n=10, words="typography body paragraph text site"),
    Side("fonts", "section_body_font", "section", "contentBlock-features", "bodyFontFamily", (
        ("focus", "change the {label} in this section to {v}"),
        ("focus", "just this section: body copy in {v}"),
        ("focus", "only here, make the paragraph font {v}"),
        ("contrast", "use {v} for the body copy in this section only, keep the rest of the site the same"),
        ("literal", "in the feature cards section, set the body copy font to {v}"),
    ), n=10, words="section body copy text feature cards"),
    # 3. names: a built-in page's header link vs its footer link vs a new custom header link vs the checkout's
    #    order-type option titles
    Side("names", "header_nav_name", "global", "HeaderSection", "pageVisibility.{page}.headerNavName", (
        ("literal", "rename {name} in the header menu to '{v}'"),
        ("literal", "in the top nav, call the {name} link '{v}' instead"),
        ("literal", "change the header link text for {name} to '{v}'"),
        ("contrast", "rename the {name} link in the header to '{v}', just the header, not the footer"),
    ), n=8, pages=("products", "contact", "faq", "locations", "loyalty"),
        values=NAV_NAMES, words="header nav link name"),
    Side("names", "footer_nav_name", "global", "FooterSection", "pageVisibility.{page}.footerNavName", (
        ("literal", "rename {name} in the footer links to '{v}'"),
        ("literal", "footer: show '{v}' instead of {name}"),
        ("contrast", "change {name} to '{v}' in the footer only, keep the header menu as it is"),
    ), n=8, pages=("products", "contact", "faq", "locations", "loyalty", "about"),
        values=NAV_NAMES, words="footer link name"),
    Side("names", "header_custom_link", "list", "HeaderSection", "pageVisibility._meta.headerNavCustomLinks", (
        ("literal", "add a custom link '{v}' to the header menu that goes to {url}"),
        ("literal", "put a new link in the top nav called '{v}' pointing to {url}"),
        ("contrast", "add '{v}' as a new link in the header nav (to {url}), don't rename the existing ones"),
    ), n=6, values=("Weekly Specials", "Events", "Careers", "Lab Results", "Gift Cards", "Order Ahead"),
        words="header nav custom link add new"),
    Side("names", "checkout_option_title", "page", "checkout", "orderTypeText.{page}.title", (
        ("literal", "on the checkout page, change the {name} option's title to '{v}'"),
        ("literal", "at checkout, rename the {name} choice to '{v}'"),
        ("contrast", "at checkout, call the {name} option '{v}', the checkout option, not a menu link"),
    ), n=8, pages=("local_delivery", "local_pickup", "mail_order"),
        values=CHECKOUT_TITLES, words="checkout title option order type"),
    # 4. grid: the product card's cards per row vs the shop page's grid columns
    Side("grid", "card_cards_per_row", "global", "ProductCardSection", "cardsPerRow", (
        ("literal", "set {label} to {v} in the product card settings"),
        ("literal", "I'd like {v} cards per row for my product cards"),
        ("contrast", "change {label} for product cards to {v}, the product card setting, not the shop page columns"),
        ("focus", "{label}: {v}"),
    ), n=10, words="product card cards per row"),
    Side("grid", "shop_grid_columns", "page", "products", "gridCols", (
        ("literal", "make the shop page grid {v} {label}"),
        ("literal", "on this page I want {v} {label} of products"),
        ("literal", "shop page: {v} {label}"),
        ("contrast", "{v} {label} on the shop page only, leave the product card setting alone"),
    ), n=10, page="products", words="shop page grid columns"),
    # 5. footer text: links + headings vs the tagline paragraph vs the link hover colour
    Side("footer", "footer_text_color", "global", "FooterSection", "footerTextColor", (
        ("literal", "make the footer headings and links {v}"),
        ("literal", "footer {label} {v}"),
        ("literal", "change the footer link text to {v}"),
        ("relative", "footer links a bit lighter please"),
        ("contrast", "set the footer link and heading colour to {v}, not the little paragraph"),
        ("focus", "{label} {v}"),
    ), n=8, words="footer headings links brand name text"),
    Side("footer", "footer_body_text_color", "global", "FooterSection", "footerBodyTextColor", (
        ("literal", "make the tagline paragraph in the footer {v}"),
        ("literal", "footer blurb text colour {v}"),
        ("relative", "the footer tagline text could be a bit darker"),
        ("contrast", "only the short paragraph in the footer should be {v}, keep the footer links as they are"),
        ("focus", "the {label} colour should be {v}"),
    ), n=8, words="footer body text tagline paragraph short"),
    Side("footer", "footer_link_hover", "global", "FooterSection", "footerLinkHighlightColor", (
        ("literal", "footer links on hover should turn {v}"),
        ("literal", "set the footer {label} to {v}"),
        ("contrast", "{v} when you hover the footer links, the normal footer link colour stays"),
    ), n=6, words="footer link highlight hover active"),
    # 6. the location picker: its headline vs the text under it
    Side("picker", "location_picker_title", "branding", "", "locationPickerTitle", (
        ("literal", "change the {label} on the location picker to '{v}'"),
        ("literal", "store picker popup headline: '{v}'"),
        ("contrast", "the big headline on the location picker should say '{v}', leave the text under it"),
    ), n=6, richtext=True, values=("Pick your store", "Where are you shopping today?", "Choose a location",
                                   "Find your nearest shop"), words="location picker headline store"),
    Side("picker", "location_picker_body", "branding", "", "locationPickerDescription", (
        ("literal", "change the {label} on the location picker to '{v}'"),
        ("literal", "the line under the location picker headline should read '{v}'"),
        ("contrast", "update the small text under the location picker headline to '{v}', the headline stays"),
    ), n=6, richtext=True, values=("Prices and stock depend on the store you pick.",
                                   "Select the shop closest to you to see what is in stock.",
                                   "Your cart follows the store you choose."), words="location picker body text"),
    # 7. the logo: desktop vs mobile max height
    Side("logo", "logo_height_desktop", "global", "BrandingSection", "logoMaxHeight", (
        ("literal", "cap the desktop logo height at {v}px"),
        ("literal", "on desktop the logo should be at most {v}px tall"),
        ("relative", "make the logo a little smaller on desktop"),
        ("contrast", "{v}px max for the logo on desktop, leave the mobile size alone"),
    ), n=6, words="desktop logo max height"),
    Side("logo", "logo_height_mobile", "global", "BrandingSection", "logoMaxHeightMobile", (
        ("literal", "on phones the logo should be at most {v}px"),
        ("literal", "mobile logo max height {v}px"),
        ("relative", "the logo is a bit big on mobile, shrink it"),
        ("contrast", "{v}px max for the logo on mobile only, the desktop size stays"),
    ), n=6, words="mobile logo max height phones"),
)
# "the font" with no part named → every site font (r4score.site_font_fields). Ids follow SIDES'.
SITE_FONT_SIDES = (
    Side("site_fonts", "site_fonts", "global", "TypographyColorsSection", "bodyFont", (
        ("literal", "use {v} for every font on the site"),
        ("literal", "switch all the fonts to {v}"),
        ("literal", "{v} for the whole site's typography please"),
        ("literal", "make every font on my site {v}"),
        ("literal", "set all the site fonts to {v}"),
        ("literal", "one font for the entire website: {v}"),
    ), n=12, words="typography font fonts site website"),
)
# a field named, nothing asked of it: the answer is a question (mode ask)
VAGUE_TEMPLATES = ("i'm not happy with the {label}", "the {label} doesn't feel right", "can we rethink the {label}",
                   "something is off about the {label}", "not sure about the {label} at the moment",
                   "the {label} needs some work")
VAGUE_GENERATOR = "vague-1"
SITE_FONT_BASE, VAGUE_BASE = 800, 900  # id numbers: contrast1-F8-0801…, contrast1-F8-0901…
# the place a global panel's field sits, in the merchant's words (a vague request names it with the label)
PLACE_WORDS = {"ShopMenuSection": "shop menu", "FooterSection": "footer", "HeaderSection": "header",
               "ProductCardSection": "product card", "BrandingSection": "logo", "ButtonsSection": "button",
               "SidebarCartSection": "side cart", "TypographyColorsSection": "site's"}
PAGE_PLACE = {"products": "shop page", "checkout": "checkout"}
ASK_FIRST_GENERATOR = "askfirst-1"

# --- the capability groups (create_page, dropdown lists) ------------------------------------------------------------------

PAGE_GENERATOR, LIST_GENERATOR = "page-1", "list-1"
PAGE_BASE, LIST_BASE = 1000, 1100  # id numbers: contrast1-F8-1001…, contrast1-F8-1101…
# a live premium template (launcher ThemeSelector: plantrepublic is on the premium allowlist; spiritleaf, smokeshow,
# plantlife and generation are hidden legacy ones; not in BASIC_TEMPLATE_IDS): create_page is refused E_TARGET. Not
# `default`, which PreviewPage treats as basic although BASIC_TEMPLATE_IDS leaves it out (the engine refuses both)
PREMIUM_TEMPLATE = "plantrepublic"
CAROUSEL_VIEW = {"tab": "editor", "activeSectionId": "productCarousel-1"}
SECTION_WORDS = {"productCarousel": "product carousel", "gallery": "photo gallery", "newsletter": "newsletter sign-up",
                 "testimonials": "testimonials section"}
# custom pages a collision / negative case adds to the state (the state already holds About Us /about and Events
# /events from fixtures/context.json — Events is never asked: exam p137's answer on the exam state)
EXTRA_PAGES = {"deals": "Deals", "our-team": "Our Team", "community": "Community", "our-partners": "Our Partners"}
# (group, request, spec): spec {title, slug?, slugs?, in_nav?, status?, sections?, why? (a decline), pages?
# (EXTRA_PAGES slugs on the state), page?, template?, where? (mode where), section? (mode keys: {page, type})}
PAGE_JOBS: tuple = (
    ("create", "add a new page called Strain Library", {"title": "Strain Library"}),
    ("create", "can you make me a page named 'Lab Results'", {"title": "Lab Results"}),
    ("create", "new page: Vendor Days", {"title": "Vendor Days"}),
    ("create", "I need a Wholesale page on the site", {"title": "Wholesale"}),
    ("create", "set up a page titled First Time Visitors", {"title": "First Time Visitors"}),
    ("create", "create a Dosing Guide page", {"title": "Dosing Guide"}),
    ("slug", "create a page called Weekly Specials at /specials", {"title": "Weekly Specials", "slug": "specials"}),
    ("slug", "add a Terpene Guide page with the address /terpenes", {"title": "Terpene Guide", "slug": "terpenes"}),
    ("slug", "make a page named Medical Patients at /medical", {"title": "Medical Patients", "slug": "medical"}),
    ("draft", "add a Holiday Hours page but keep it as a draft for now", {"title": "Holiday Hours", "status": "draft"}),
    ("draft", "make a draft page called Price Match, don't publish it yet", {"title": "Price Match", "status": "draft"}),
    ("draft", "create a Store Policies page as a draft", {"title": "Store Policies", "status": "draft"}),
    ("in_nav", "create a Rewards Club page and put it in the header menu", {"title": "Rewards Club", "in_nav": True}),
    ("in_nav", "add a page called Gift Guide and link it in the top navigation", {"title": "Gift Guide", "in_nav": True}),
    ("in_nav", "make an Accessibility page and add it to the header nav", {"title": "Accessibility", "in_nav": True}),
    ("sections", "create a Budtender Picks page with a product carousel on it",
     {"title": "Budtender Picks", "sections": ["productCarousel"]}),
    ("sections", "make a new page called Photo Wall with a photo gallery", {"title": "Photo Wall", "sections": ["gallery"]}),
    ("sections", "add a Stay In Touch page and put a newsletter sign-up on it",
     {"title": "Stay In Touch", "sections": ["newsletter"]}),
    ("sections", "create a Customer Reviews page with a testimonials section",
     {"title": "Customer Reviews", "sections": ["testimonials"]}),
    ("exists", "make me a Deals page", {"title": "Deals", "pages": ["deals"]}),
    ("exists", "set up an Our Team page", {"title": "Our Team", "pages": ["our-team"]}),
    ("exists", "create a FAQ page", {"title": "FAQ"}),
    ("exists", "add a Contact page to the site", {"title": "Contact"}),
    ("exists_sections", "create a Deals page with a product carousel on it",
     {"title": "Deals", "pages": ["deals"], "sections": ["productCarousel"]}),
    ("exists_sections", "add a new Contact page with a newsletter sign-up", {"title": "Contact", "sections": ["newsletter"]}),
    ("exists_sections", "make a FAQ page and put a photo gallery on it", {"title": "FAQ", "sections": ["gallery"]}),
    ("reserved", "make a page called Menu", {"title": "Menu", "why": "reserved"}),
    ("reserved", "create a Sitemap page", {"title": "Sitemap", "why": "reserved"}),
    ("reserved", "add a page named Launcher", {"title": "Launcher", "why": "reserved"}),
    ("noncanonical", "create a page called Summer Sale at /Summer-Sale", {"title": "Summer Sale", "asked_slug": "Summer-Sale"}),
    ("noncanonical", "add a Wholesale Inquiries page with the address /Wholesale_Inquiries",
     {"title": "Wholesale Inquiries", "asked_slug": "Wholesale_Inquiries"}),
    ("noncanonical", "make a Local Growers page at /Local Growers", {"title": "Local Growers", "asked_slug": "Local Growers"}),
    ("template", "add a new page called Gift Cards", {"title": "Gift Cards", "why": "template"}),
    ("template", "create a Lab Results page and put it in the menu", {"title": "Lab Results", "why": "template", "in_nav": True}),
    ("manage", "delete the Deals page", {"where": "pages", "pages": ["deals"]}),
    ("manage", "duplicate the About Us page", {"where": "pages"}),
    ("manage", "change the address of the Our Team page to /crew", {"where": "pages", "pages": ["our-team"]}),
    ("manage", "rename the Community page to Neighbours", {"where": "pages", "pages": ["community"]}),
    ("section_not_page", "add a product carousel to the Our Partners page",
     {"section": {"page": "our-partners", "type": "productCarousel"}, "pages": ["our-partners"]}),
    ("section_not_page", "put a testimonials section on my Community page",
     {"section": {"page": "community", "type": "testimonials"}, "pages": ["community"]}),
    ("section_not_page", "on the Contact page, add a newsletter sign-up", {"section": {"page": "contact", "type": "newsletter"}}),
    ("section_not_page", "add a photo gallery to the delivery page", {"section": {"page": "delivery", "type": "gallery"}}),
    # "our <network> page" is a social link (SocialMediaSection), never a new site page (exam h022 "add our weedmaps
    # page" is one; the exam's own networks — weedmaps, instagram, facebook, tiktok — are left to it)
    ("social_page", "add our LinkedIn page: https://www.linkedin.com/company/example-dispensary",
     {"social": {"key": "socialLinks.linkedin", "url": "https://www.linkedin.com/company/example-dispensary"}}),
    ("social_page", "link our Leafly page https://www.leafly.com/dispensary-info/example-dispensary",
     {"social": {"key": "socialLinks.leafly", "url": "https://www.leafly.com/dispensary-info/example-dispensary"}}),
    ("social_page", "put our Google Business page in the footer: https://g.page/example-dispensary",
     {"social": {"key": "socialLinks.googleBusiness", "url": "https://g.page/example-dispensary"}}),
    ("social_page", "add our X page too, https://x.com/exampledispensary",
     {"social": {"key": "socialLinks.twitter", "url": "https://x.com/exampledispensary"}}),
)
# (a section add's write_ids — <page>.sections, <page>.sections[].id, <page>.id — must not be a generator hold-out key:
# home, faq, deals, reviews… are, so the "add a section" pages are ones that are not)
# tag / brand / category names (made up; never a real store's): A × B, no name a whole-word part of another
TAG_A = ("Small Batch", "Staff", "Local", "Limited", "Weekend", "Budget", "Premium", "Craft", "Seasonal", "Classic",
         "Fresh", "Rare", "Mellow", "Bright", "Quiet", "Golden", "Late Night", "Morning", "Holiday", "Sunday")
TAG_B = ("Picks", "Favourites", "Finds", "Deals", "Drops", "Classics", "Specials", "Essentials")
BRAND_A = ("Cedar", "Harbor", "Prairie", "Northwind", "Copper", "Juniper", "Maple", "Riverbend", "Stonefield",
           "Bluebird", "Foxglove", "Lantern", "Meadowlark", "Ironwood", "Silverleaf", "Thistle")
BRAND_B = ("Growers", "Farms", "Botanicals", "Collective", "Gardens", "Leaf Co", "Craft", "Supply")
MANY_CATEGORIES = ("Flower", "Pre-Rolls", "Edibles", "Vapes", "Concentrates", "Accessories", "Beverages", "Topicals",
                   "Capsules", "Oils", "Seeds", "Hash", "Rosin", "Gummies", "Chocolates", "Tinctures", "Papers",
                   "Grinders", "Lighters", "Storage", "Apparel", "Gift Cards")
# (request, list, situation, size[, focus]): size = how many the store has (None = the fixture's own list; "unknown" =
# the browser could not load it: the case's context leaves the member out); focus "plain" = no focused carousel (a
# general "what brands do we carry" asked on home; the request's own words open the tag / brand gate) — default: a
# focused product carousel. Sizes straddle the rules' edges: 15 (named in full) / 16 (15 then "...and 1 more"), 99 and
# 100 (not cut) / 101 (cut, the dropdown named). The made-up banks hold 160 tags and 128 brands
LIST_JOBS: tuple = (
    ("which tags can I use for this carousel?", "tags", "short", 6),
    ("what brands can this carousel show?", "brands", "short", 5),
    ("what categories can I pick for the carousel?", "categories", "short", None),
    ("list the tags I can choose from for this product carousel", "tags", "long", 24),
    ("show me the brand options for this carousel", "brands", "long", 40),
    ("what tags are there for the carousel source?", "tags", "truncated", 130),
    ("which brands could I feature in this carousel?", "brands", "truncated", 120),
    ("what tags can I filter this carousel by?", "tags", "empty", 0),
    ("can you list the brands for this carousel?", "brands", "empty", 0),
    ("what are my tag options for this carousel?", "tags", "unknown", None),
    ("which brands are available for the carousel?", "brands", "unknown", None),
    ("list every category I could use for this carousel", "categories", "long", 22),
    # r6-4 review: more of every situation, varied totals, and requests that are not about a carousel
    ("what tags do I have set up?", "tags", "short", 3, "plain"),
    ("which brands can I pick from?", "brands", "short", 1, "plain"),
    ("show me all the tags for this carousel", "tags", "short", 15),
    ("what brands could this product carousel use?", "brands", "short", 9),
    ("what product categories does my store have?", "categories", "short", 8, "plain"),
    ("which categories can this carousel pull from?", "categories", "short", 12),
    ("list the brands I can use", "brands", "long", 16, "plain"),
    ("what tags are available for the carousel?", "tags", "long", 33),
    ("which brands do we carry?", "brands", "long", 57, "plain"),
    ("give me the tag options for this carousel", "tags", "long", 99),
    ("what are all my product tags?", "tags", "long", 100, "plain"),
    ("what categories could I feature here?", "categories", "long", 20),
    ("which tags can I choose from?", "tags", "truncated", 101, "plain"),
    ("list every brand for this carousel", "brands", "truncated", 128),
    ("what tags could this carousel filter on?", "tags", "truncated", 160),
    ("show me the brands we have", "brands", "truncated", 110, "plain"),
    ("do I have any brands set up?", "brands", "empty", 0, "plain"),
    ("what tags exist in my store?", "tags", "empty", 0, "plain"),
    ("which categories can I use for this carousel?", "categories", "empty", 0),
    ("are there any tags I could use on this carousel?", "tags", "empty", 0),
    ("what brands can I pick from?", "brands", "unknown", None, "plain"),
    ("list my tags", "tags", "unknown", None, "plain"),
    ("which tags could I feature in this carousel?", "tags", "unknown", None),
    ("what brand options does this carousel have?", "brands", "unknown", None),
)


def _names(a: tuple, b: tuple, n: int) -> list[str]:
    """`n` names "<a> <b>", spread over both banks (a fixed hash order)."""
    return sorted((f"{x} {y}" for x in a for y in b), key=lambda v: (h8("name:" + v), v))[:n]


def list_entries(lst: str, n: int, salt: str) -> list[dict]:
    """`n` made-up entries {slug, name, id} of a tag / brand / category list (deterministic)."""
    if lst == "categories":
        names = list(MANY_CATEGORIES[:n])
    else:
        names = _names(TAG_A, TAG_B, n) if lst == "tags" else _names(BRAND_A, BRAND_B, n)
    from .exam import slugify
    return [{"slug": slugify(x), "name": x, "id": str(100 + i)} for i, x in enumerate(names)]


def _with_pages(state: dict, slugs: list[str]) -> None:
    pages = list(state.get("customPages") or [])
    for sl in slugs or []:
        if not any(p.get("slug") == sl for p in pages):
            pages.append({"slug": sl, "title": EXTRA_PAGES[sl], "sections": [], "settings": {}, "status": "published"})
    state["customPages"] = pages


def _empty_expected() -> dict:
    return {"ops": [], "diff": [], "keys": [], "alt": [], "alt_exact": [], "out_of_scope": None, "candidates": None,
            "direction": None, "reply_any": None, "expect_hidden": None, "gate": None}


def build_page_case(m, group: str, request: str, spec: dict, case_id: str) -> tuple[dict | None, str]:
    """One create_page-family case (PAGE_JOBS): state, request, expectation, oracle ops (and, for a decline, the
    create_page the engine must refuse: meta.attempt). Measured by `validate`."""
    from .exam import slugify
    from .state import build_state
    salt = f"{PAGE_GENERATOR}:{case_id}"
    tpl = TEMPLATES[h8("tpl:" + salt) % len(TEMPLATES)]
    state = build_state(m, template=tpl)
    _with_pages(state, spec.get("pages") or [])
    if spec.get("why") == "template":
        tpl = PREMIUM_TEMPLATE
        state["template"] = {"id": tpl}
    exp = _empty_expected()
    meta_extra: dict[str, Any] = {}
    oracle: list[dict] = []
    if spec.get("where"):
        mode, kind = "where", "unsupported"
        exp["out_of_scope"] = {"reason": "pages are renamed, duplicated and deleted in the Pages tab",
                               "where": [spec["where"]], "expected_where": [spec["where"]]}
        exp["where"] = [spec["where"]]
    elif spec.get("social"):
        mode, kind = "keys", "contrast"
        so = spec["social"]
        tgt = {"kind": "global", "panel": "SocialMediaSection"}
        oracle = [{"op": "set_fields", "target": tgt, "values": {so["key"]: so["url"]}}]
        exp["ops"] = copy.deepcopy(oracle)
        exp["keys"] = [{"target": tgt, "key": so["key"], "list": ""}]
    elif spec.get("section"):
        mode, kind = "keys", "contrast"
        sec = spec["section"]
        tgt = {"kind": "page", "page": sec["page"]}
        oracle = [{"op": "edit_list", "target": tgt, "list": "sections", "action": "add", "item": {"id": sec["type"]}}]
        exp["ops"] = copy.deepcopy(oracle)
        exp["keys"] = [{"target": tgt, "key": "", "list": "sections"}]
        exp["list_ops"] = [{"list": "sections", "action": "add", "index": None}]
    else:
        mode, kind = "page", "create_page"
        title = spec["title"]
        page: dict[str, Any] = {"expect": "decline" if spec.get("why") in ("reserved", "template") else "create",
                                "title": title, "title_match": "exact", "in_nav": bool(spec.get("in_nav")),
                                "status": spec.get("status") or "published", "sections": list(spec.get("sections") or [])}
        if page["expect"] == "decline":
            page["why"] = spec["why"]
        else:
            page["if_exists"] = "decline"
        asked = spec.get("asked_slug")
        if spec.get("slug"):
            page["slug"] = spec["slug"]
        if asked:
            page["slugs"] = sorted({slugify(asked), slugify(re.sub(r"[_./]+", " ", asked))} - {""})
        slug = spec.get("slug") or (page["slugs"][0] if asked else slugify(title))
        op: dict[str, Any] = {"op": "create_page", "title": title}
        if spec.get("slug") or asked or page["sections"]:
            op["slug"] = slug
        if page["in_nav"]:
            op["in_nav"] = True
        if page["status"] != "published":
            op["status"] = page["status"]
        adds = [{"op": "edit_list", "target": {"kind": "page", "page": slug}, "list": "sections", "action": "add",
                 "item": {"id": t}} for t in page["sections"]]
        exp["page"] = page
        from .exam import existing_page
        exists = existing_page({"site_state": state, "context": {}}, None, title, [slug])
        if page["expect"] == "decline" or exists:
            meta_extra["attempt"] = [op] + adds  # what the engine must refuse (the create; adds land on nothing)
        else:
            oracle = [op] + adds
            exp["ops"] = copy.deepcopy(oracle)
    case = {"id": case_id, "kind": kind, "target": {"kind": "page", "page": "home"}, "fields": [], "site_state": state,
            "context": {"page": "home"}, "request": request, "history": [], "expected": exp, "score": {"mode": mode},
            "holdout": False, "manifest_hash": m.hash, "engine_fingerprint": None, "template": tpl,
            "meta": {"family": FAMILY, "style": group, "page": "home", "view": None, "oracle_ops": copy.deepcopy(oracle),
                     # the split's unit (r4dataset: h100(group) < val_pct → val) is the CASE: a shared group name
                     # sent a whole behaviour class to one side ("page:exists" all to val); cap_group reports it
                     "group": f"page:{group}:{case_id}", "cap_group": f"page:{group}", "capability": "create_page",
                     "generator": PAGE_GENERATOR,
                     "contrast": {"pair": "create_page", "side": group, "key": None, "page_key": None, "value": None,
                                  "others": []}, **meta_extra}}
    return case, ""


def build_list_case(m, request: str, lst: str, situation: str, size: int | None, case_id: str,
                    focus: str = "carousel") -> tuple[dict | None, str]:
    """One list_options case (LIST_JOBS): a focused product carousel (focus "plain": nothing focused, the home page),
    the store's list in `option_lists` (tags, brands; the other kind is the fixture's) or `context_extras`
    (categories)."""
    from .state import build_state
    salt = f"{LIST_GENERATOR}:{case_id}"
    tpl = TEMPLATES[h8("tpl:" + salt) % len(TEMPLATES)]
    state = build_state(m, template=tpl)
    from .state import load_context_fixture
    fx = load_context_fixture()
    extras: dict[str, Any] = {}
    option_lists = None
    mine = None if situation == "unknown" else (list_entries(lst, size, salt) if size is not None else None)
    if lst in ("tags", "brands"):  # the host's carousel lists (state.option_lists_of): the other kind as the fixture's
        option_lists = {k: copy.deepcopy(fx.get(k)) for k in ("tags", "brands")}
        option_lists[lst] = mine
    elif mine is not None:
        extras[lst] = mine
    exp = _empty_expected()
    exp["listing"] = {"list": lst, "situation": situation}
    plain = focus == "plain"
    target = {"kind": "page", "page": "home"} if plain else {"kind": "section", "page": "home", "sectionId": "productCarousel-1"}
    case = {"id": case_id, "kind": "answer", "target": target,
            "fields": [], "site_state": state, "context": {"page": "home"}, "request": request, "history": [],
            "expected": exp, "score": {"mode": "list"}, "holdout": False, "manifest_hash": m.hash,
            "engine_fingerprint": None, "template": tpl, "context_extras": extras,
            **({"option_lists": option_lists} if option_lists is not None else {}),
            "meta": {"family": FAMILY, "style": situation, "page": "home", "view": None if plain else dict(CAROUSEL_VIEW),
                     "oracle_ops": [], "focus": focus,
                     "group": f"list:{lst}:{case_id}", "cap_group": f"list:{lst}", "capability": "list_options",
                     "generator": LIST_GENERATOR,
                     "contrast": {"pair": "list_options", "side": f"{lst}:{situation}", "key": None, "page_key": None,
                                  "value": None, "others": []}}}
    return case, ""


# The capability REPORT set (eval_stage `capability:built`, never bar): ~15 cases built the same way in their own id
# space (caph-F8-nnnn), in phrasings the training jobs do not use, so a candidate that over-creates pages, invents
# list names or calls an unknown list empty shows in the report. Built at eval time on the deployed manifest, scored by
# r4score like the training cases; GPU time only. They are no hold-out (the training groups teach the same skills).
CAPH_PAGE_JOBS: tuple = (
    ("create", "please add a page called Grower Spotlights", {"title": "Grower Spotlights"}),
    ("slug", "make a Pet Friendly page at /pets", {"title": "Pet Friendly", "slug": "pets"}),
    ("draft", "start a draft page called Fall Lineup", {"title": "Fall Lineup", "status": "draft"}),
    ("in_nav", "add a Neighbourhood News page to the header menu", {"title": "Neighbourhood News", "in_nav": True}),
    ("sections", "create a Staff Favourites page with a product carousel",
     {"title": "Staff Favourites", "sections": ["productCarousel"]}),
    ("exists", "create an About Us page", {"title": "About Us"}),
    ("exists", "make a Careers page", {"title": "Careers"}),
    ("reserved", "add a page named API", {"title": "API", "why": "reserved"}),
    ("template", "create a page called Gift Ideas", {"title": "Gift Ideas", "why": "template"}),
    ("section_not_page", "add a newsletter sign-up to the About Us page", {"section": {"page": "about", "type": "newsletter"}}),
)
CAPH_LIST_JOBS: tuple = (
    ("what tags have I got?", "tags", "short", 4, "plain"),
    ("what brands can this carousel use?", "brands", "long", 30),
    ("list the tags for this carousel", "tags", "truncated", 140),
    ("do we have any brands?", "brands", "empty", 0, "plain"),
    ("what tags can I use here?", "tags", "unknown", None),
)


def capability_report_cases(m, *, limit: int | None = None) -> list[dict]:
    """The capability report set (CAPH_PAGE_JOBS + CAPH_LIST_JOBS) on manifest `m`, unmeasured (eval scores them as
    built; on an engine without create_page its creates fail on both arms alike)."""
    out: list[dict] = []
    for k, (group, request, spec) in enumerate(CAPH_PAGE_JOBS):
        c, _ = build_page_case(m, group, request, spec, f"caph-{FAMILY}-{PAGE_BASE + k + 1:04d}")
        if c is not None:
            out.append(c)
    for k, job in enumerate(CAPH_LIST_JOBS):
        c, _ = build_list_case(m, *job[:4], f"caph-{FAMILY}-{LIST_BASE + k + 1:04d}", *job[4:])
        if c is not None:
            out.append(c)
    for c in out:
        c["meta"]["report_set"] = "capability"
    return out[:limit] if limit else out


def page_exam_clash(c: dict) -> str | None:
    """A page case that teaches an exam row's page answer (exam.AMENDMENTS page specs: p137 "Events"): its title's
    slug is the exam row's."""
    from .exam import AMENDMENTS, slugify
    page = (c.get("expected") or {}).get("page") or {}
    meta = c.get("meta") or {}
    ops = list(meta.get("attempt") or []) + list(meta.get("oracle_ops") or []) + list((c.get("expected") or {}).get("ops") or [])
    mine = {slugify(page.get("title") or "")} | {slugify(w) for w in page.get("words") or []} \
        | {o.get("slug") or slugify(o.get("title") or "") for o in ops if isinstance(o, dict) and o.get("op") == "create_page"}
    for (s, rid), a in AMENDMENTS.items():
        t = (a.get("page") or {}).get("title")
        if t and slugify(t) in mine - {""}:
            return f"exam_answer:{s}:{rid}:page"
    return None


# --- metadata ---------------------------------------------------------------------------------------------------------

def _words(s: Any) -> set[str]:
    """Content words, a plural folded onto its singular."""
    return {(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
            for w in re.findall(r"[a-z]+", str(s or "").lower()) if w not in STOP and len(w) > 2}


def label_of(f) -> str:
    """The merchant-facing label, lower case; a "Group › detail" label keeps the group (what the editor shows)."""
    lab = str(f.label or "").split("›")[0].strip()
    return re.sub(r"\s+", " ", lab).lower()


def container_of(m, side: Side):
    if side.kind in ("global", "list"):
        return m.panel(side.container)
    if side.kind == "page":
        return m.page(side.container)
    if side.kind == "branding":
        return m.branding
    if side.kind == "section":
        return m.section(side.container)
    raise ValueError(side.kind)


def target_of(side: Side) -> dict:
    if side.kind in ("global", "list"):
        return {"kind": "global", "panel": side.container}
    if side.kind == "page":
        return {"kind": "page", "page": side.container}
    if side.kind == "branding":
        return {"kind": "branding"}
    return {"kind": "section", "page": "home", "sectionId": side.container}


def nav_name(m, panel: str, page: str) -> str | None:
    """A built-in page's display name from its visibility toggle's manifest label ("Hide Shop from header nav",
    "Show FAQ in footer") — None when the label is a placeholder."""
    c = m.panel(panel)
    if c is None:
        return None
    for k in (f"pageVisibility.{page}.inHeaderNav", f"pageVisibility.{page}.inFooterNav"):
        f = c.fields.get(k)
        if f is None:
            continue
        mt = re.match(r"^(?:Hide|Show)\s+(.+?)\s+(?:from|in)\s+(?:the\s+)?(?:header|footer)", str(f.label or ""))
        if mt and "{" not in mt.group(1):
            return mt.group(1)
    return None


def option_name(f) -> str | None:
    """"Title (Delivery)" → "delivery"."""
    mt = re.search(r"\(([^)]+)\)", str(f.label or ""))
    return mt.group(1).lower() if mt else None


def check_templates(m, sides: tuple[Side, ...] = SIDES) -> list[str]:
    """Problems: a side whose field is missing, or a template that neither fills {label} nor shares a content word
    with the field's label, help, list item labels or the side's container words."""
    out = []
    for s in sides:
        c = container_of(m, s)
        if c is None:
            out.append(f"{s.name}: no container {s.kind}:{s.container}")
            continue
        keys = [s.key.replace("{page}", p) for p in s.pages] if s.pages else [s.key]
        meta_words = _words(s.words) | _words(c.name)
        for k in keys:
            if s.kind == "list":
                lst = c.lists.get(k)
                if lst is None:
                    out.append(f"{s.name}: no list {k}")
                    continue
                meta_words |= _words(k) | {w for f in lst.item_fields.values() for w in _words(f.label)}
                continue
            f = c.fields.get(k)
            if f is None:
                out.append(f"{s.name}: no field {k}")
                continue
            meta_words |= _words(f.label) | _words(f.spec.get("help")) | _words(re.sub(r"([A-Z])", r" \1", k))
        for style, t in s.templates:
            if "{label}" in t:
                continue
            if not (_words(t.replace("{v}", "").replace("{name}", "").replace("{url}", "")) & meta_words):
                out.append(f"{s.name}: template shares no metadata word: {t!r}")
    return out


# --- values -------------------------------------------------------------------------------------------------------------

def _hex(v: Any) -> str | None:
    s = str(v or "").strip().lower()
    return s if re.fullmatch(r"#[0-9a-f]{6}", s) else None


def shift_color(hexv: str, darker: bool) -> str:
    h = hexv.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    rgb = [int(x * 0.7) for x in rgb] if darker else [int(x + (255 - x) * 0.35) for x in rgb]
    return "#" + "".join(f"{x:02x}" for x in rgb)


def pick_value(f, current: Any, salt: str, bank: tuple = ()) -> Any:
    """A literal value of the field's type that differs from `current`."""
    rank = h8(salt)
    if bank:
        opts = [b for b in bank if b != current]
        return opts[rank % len(opts)]
    t = f.type
    if t == "color":
        opts = [p for p in PALETTE if p != str(current or "").lower()]
        return opts[rank % len(opts)]
    if t in ("select", "enum"):
        opts = [e for e in f.enum if e not in (None, "", current) and e not in FONT_SKIP]
        return opts[rank % len(opts)] if opts else None
    if t in ("range", "number"):
        lo, hi = int(f.min if f.min is not None else 10), int(f.max if f.max is not None else 40)
        opts = [x for x in range(lo, hi + 1) if x != current and (x % 2 == 0 or hi - lo < 12)]
        return opts[rank % len(opts)] if opts else None
    return None


def typed(v: Any) -> str:
    return str(v)


# --- the cases ----------------------------------------------------------------------------------------------------------

def _seed(m, state: dict, seeds: tuple) -> None:
    from .state import write_field
    for panel, key, value in seeds:
        c = m.panel(panel)
        if c is not None and key in c.fields:
            write_field(c, c.fields[key], state, value)


def _current(m, side: Side, state: dict, key: str) -> Any:
    from .state import read_field
    c = container_of(m, side)
    f = c.fields.get(key) if c else None
    if f is None:
        return None
    try:
        return read_field(c, f, state, section_id=side.container if side.kind == "section" else None)
    except (KeyError, ValueError):
        return None


def _write(m, side: Side, state: dict, key: str, value: Any) -> None:
    from .state import write_field
    c = container_of(m, side)
    write_field(c, c.fields[key], state, value, section_id=side.container if side.kind == "section" else None)


def build_case(m, side: Side, i: int, style: str, template: str, case_id: str) -> tuple[dict | None, str]:
    """(case, "") or (None, why). The state, the request, the oracle ops and the expectation of one case (not yet
    measured: `validate` does that)."""
    from .state import build_state
    salt = f"{GENERATOR}:{side.name}:{i}"
    tpl = TEMPLATES[h8("tpl:" + salt) % len(TEMPLATES)]
    state = build_state(m, template=tpl)
    _seed(m, state, side.seed)
    c = container_of(m, side)
    page_key = side.pages[h8("page:" + salt) % len(side.pages)] if side.pages else None
    key = side.key.replace("{page}", page_key) if page_key else side.key
    bank = side.values
    if bank and isinstance(bank[0], tuple):
        bank = dict(bank).get(page_key) or ()
    target = target_of(side)
    name = None
    if "{page}" in side.key and side.kind == "global":
        name = nav_name(m, side.container, page_key)
        if not name:
            return None, "no_display_name"
    if side.kind == "page" and "{page}" in side.key:
        f0 = c.fields.get(key) if c else None
        name = option_name(f0) if f0 else None
        if not name:
            return None, "no_option_name"
    view = None
    if style == "focus":
        view = ({"tab": "editor", "activeSectionId": side.container} if side.kind == "section" else
                {"tab": "settings", "globalSection": PANEL_SECTION.get(side.container, side.container)})
    elif side.kind == "section":
        view = {"tab": "editor", "activeSectionId": side.container}
    exp: dict[str, Any] = {"ops": [], "diff": [], "keys": [], "alt": [], "alt_exact": [], "out_of_scope": None,
                           "candidates": None, "direction": None, "reply_any": None, "expect_hidden": None, "gate": None}
    mode = "exact"
    url = None
    if side.kind == "list":
        lst = c.lists.get(key) if c else None
        if lst is None:
            return None, "no_list"
        v = pick_value(None, None, salt, bank)
        url = "/" + re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")
        item = {"label": v, "url": url}
        ops = [{"op": "edit_list", "target": target, "list": key, "action": "add", "item": item}]
        exp["ops"] = ops
        exp["keys"] = [{"target": target, "key": "", "list": key}]
        exp["list_ops"] = [{"list": key, "action": "add", "index": None}]
        mode = "keys"
        text = template.format(v=v, url=url, label="", name="")
        fields = [key]
    else:
        f = c.fields.get(key) if c else None
        if f is None:
            return None, "no_field"
        label = label_of(f)
        cur = _current(m, side, state, key)
        if style == "relative":
            if f.type == "color":
                if not _hex(cur):
                    cur = PALETTE[h8("seed:" + salt) % len(PALETTE)]
                    _write(m, side, state, key, cur)
                darker = bool(re.search(r"darker|deeper", template))
                v = shift_color(_hex(cur), darker)
                direction = "darker" if darker else "lighter"
            elif f.type in ("range", "number"):
                lo, hi = int(f.min if f.min is not None else 10), int(f.max if f.max is not None else 40)
                if not isinstance(cur, (int, float)) or isinstance(cur, bool):
                    cur = lo + (hi - lo) // 2
                    _write(m, side, state, key, cur)
                up = bool(re.search(r"bigger|bump|up\b|larger|taller", template))
                v = min(hi, int(cur) + 2) if up else max(lo, int(cur) - 2)
                direction = "up" if up else "down"
            else:
                return None, "relative_needs_number_or_color"
            if v == cur:
                return None, "relative_no_room"
            ops = [{"op": "set_fields", "target": target, "values": {key: v}}]
            exp["ops"] = ops
            exp["keys"] = [{"target": target, "key": key, "list": ""}]
            exp["direction"] = {"key": key, "dir": direction, "from": cur}
            mode = "keys"
            text = template.format(v="", label=label, name=name or "", url="")
        else:
            v = pick_value(f, cur, salt, bank)
            if v is None:
                return None, "no_value"
            stored = f"<p>{v}</p>" if side.richtext else v
            ops = [{"op": "set_fields", "target": target, "values": {key: stored}}]
            exp["ops"] = ops
            exp["keys"] = [{"target": target, "key": key, "list": ""}]
            text = template.format(v=typed(v), label=label, name=name or "", url="")
            mode = "keys" if side.richtext else "exact"
        fields = [key]
    text = re.sub(r"\s+", " ", text).strip()
    case = {"id": case_id, "kind": "contrast", "target": target, "fields": fields, "site_state": state,
            "context": {"page": side.page}, "request": text, "history": [], "expected": exp,
            "score": {"mode": mode}, "holdout": False, "manifest_hash": m.hash, "engine_fingerprint": None,
            "template": tpl,
            "meta": {"family": FAMILY, "style": style, "page": side.page, "view": view,
                     "oracle_ops": copy.deepcopy(exp["ops"]), "group": f"contrast:{side.pair}:{side.name}",
                     "contrast": {"pair": side.pair, "side": side.name, "key": key, "page_key": page_key,
                                  "value": None if side.kind != "list" and style == "relative" else v,
                                  "others": [s.name for s in SIDES if s.pair == side.pair and s.name != side.name]},
                     "generator": GENERATOR, "template_text": template}}
    if side.richtext and mode == "keys":
        case["meta"]["exact_to_keys"] = ["contrast_richtext"]
    if side.pair == "site_fonts":
        from .r4score import to_all_fonts
        case = to_all_fonts(case, m, v)
        case["meta"]["contrast"]["keys"] = list(case["expected"]["site_fonts"]["required"])
        case["fields"] = list(case["expected"]["site_fonts"]["required"])
    return case, ""


def build_vague_case(m, side: Side, template: str, case_id: str) -> tuple[dict | None, str]:
    """A request that names `side`'s field by its manifest label and asks nothing of it: mode ask."""
    from .state import build_state
    if side.kind not in ("global", "page", "branding", "section") or "{page}" in side.key:
        return None, "vague_unsupported_side"
    c = container_of(m, side)
    f = c.fields.get(side.key) if c else None
    if f is None:
        return None, "no_field"
    label = label_of(f)
    if not label or "{" in label:
        return None, "no_label"
    place = (PLACE_WORDS.get(side.container, "") if side.kind == "global" else
             "location picker" if side.kind == "branding" and "picker" in side.words else
             PAGE_PLACE.get(side.container, "") if side.kind == "page" else "")  # a section: its editor is open
    if place and not (_words(place) & _words(label)):
        label = f"{place} {label}"  # "the font" alone is every font (r4score.site_font_fields): name the place
    salt = f"{VAGUE_GENERATOR}:{side.name}"
    tpl = TEMPLATES[h8("tpl:" + salt) % len(TEMPLATES)]
    state = build_state(m, template=tpl)
    _seed(m, state, side.seed)
    view = {"tab": "editor", "activeSectionId": side.container} if side.kind == "section" else None  # "this section"
    text = re.sub(r"\s+", " ", template.format(label=label)).strip()
    exp = {"ops": [], "diff": [], "keys": [], "alt": [], "alt_exact": [], "out_of_scope": None, "candidates": None,
           "direction": None, "reply_any": None, "expect_hidden": None, "gate": None}
    case = {"id": case_id, "kind": "ask", "target": target_of(side), "fields": [side.key], "site_state": state,
            "context": {"page": side.page}, "request": text, "history": [], "expected": exp,
            "score": {"mode": "ask"}, "holdout": False, "manifest_hash": m.hash, "engine_fingerprint": None,
            "template": tpl,
            "meta": {"family": FAMILY, "style": "vague", "page": side.page, "view": view, "oracle_ops": [],
                     "group": f"vague:{side.name}",
                     "contrast": {"pair": "vague", "side": side.name, "key": side.key, "page_key": None, "value": None,
                                  "others": []},
                     "generator": VAGUE_GENERATOR, "template_text": template}}
    return case, ""


def ask_first_cases(pool: list[dict], m, engine, extras: dict, *, leak_index=None, seen: set[str] | None = None,
                    limit: int | None = None) -> tuple[list[dict], dict]:
    """The follow-up family's vague first turns as single-turn `ask` cases (module docstring): a trainable pool
    case whose history[0] the author marked `ask` or `offer` gives its first user text on its own state, page and
    view, scored ask (no write, a question). Same validation as a generated case (contexts derived, leak with the
    near-exam margin, dedup against the pool)."""
    from .r4cases import dedup_key, remeasure
    out: list[dict] = []
    dropped: Counter = Counter()
    for c in sorted(pool, key=lambda x: str(x.get("id"))):
        hist = c.get("history") or []
        if c.get("holdout") or not hist or str(hist[0].get("expect") or "") not in ("ask", "offer"):
            continue
        if limit is not None and len(out) >= limit:
            dropped["limit"] += 1
            continue
        h0 = hist[0]
        meta = c.get("meta") or {}
        fam = str(meta.get("family") or "F2")
        page = h0.get("page") or meta.get("page") or (c.get("context") or {}).get("page") or "home"
        new = {"id": f"{ASK_FIRST_GENERATOR.replace('-', '')}-{fam}-{c.get('id')}", "kind": "ask",
               "target": c.get("target"), "fields": list(c.get("fields") or []),
               "site_state": copy.deepcopy(c.get("site_state") or {}), "context": {"page": page},
               "request": str(h0.get("user") or ""), "history": [],
               "expected": {"ops": [], "diff": [], "keys": [], "alt": [], "alt_exact": [], "out_of_scope": None,
                            "candidates": None, "direction": None, "reply_any": None, "expect_hidden": None},
               "score": {"mode": "ask"}, "holdout": False, "manifest_hash": c.get("manifest_hash") or m.hash,
               "engine_fingerprint": None, "template": c.get("template"),
               "meta": {"family": fam, "page": page, "view": meta.get("view"), "group": meta.get("group") or c.get("id"),
                        "oracle_ops": [], "generator": ASK_FIRST_GENERATOR,
                        "ask_first": {"from": c.get("id"), "expect": h0.get("expect"),
                                      "assistant": h0.get("assistant")}}}
        if not new["request"].strip():
            dropped["empty"] += 1
            continue
        err = remeasure(new, engine, m, extras)
        if err:
            dropped[f"oracle:{err}"] += 1
            continue
        new["engine_fingerprint"] = getattr(engine, "fingerprint", None)
        if leak_index is not None:
            res = leak_index.check(new)
            if res["rule1"] or res["rule2"] or res["rule3"] or res["max_jaccard"] >= NEAR_EXAM:
                dropped["leak"] += 1
                continue
            from .leak import CHECKER
            new["meta"]["leak"] = {"max_jaccard": res["max_jaccard"], "nearest": res["nearest"], "checker": CHECKER}
        if seen is not None:
            k = dedup_key(new)
            if k in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(k)
        out.append(new)
    rep = {"generator": ASK_FIRST_GENERATOR, "kept": len(out), "dropped": dict(dropped),
           "by_family": dict(Counter(c["meta"]["family"] for c in out)),
           "by_expect": dict(Counter(c["meta"]["ask_first"]["expect"] for c in out))}
    return out, rep


def capability_counts(cases: list[dict]) -> dict:
    """{capability: {cases, by_group: {group: n}, modes: {mode: n}}} of the cases tagged meta.capability (the
    create_page and list_options groups, prepare's relabelled page-create rows)."""
    out: dict[str, dict] = {}
    for c in cases:
        meta = c.get("meta") or {}
        cap = meta.get("capability")
        if not cap:
            continue
        rec = out.setdefault(str(cap), {"cases": 0, "by_group": Counter(), "modes": Counter()})
        rec["cases"] += 1
        rec["by_group"][str(meta.get("cap_group") or meta.get("group") or "?")] += 1
        rec["modes"][str((c.get("score") or {}).get("mode"))] += 1
    return {k: {"cases": v["cases"], "by_group": dict(sorted(v["by_group"].items())), "modes": dict(v["modes"])}
            for k, v in sorted(out.items())}


PER_PAGE_KEY = re.compile(r"^(pageVisibility|orderTypeText)\.([^.]+)\.(.+)$")


def _key_shape(key: str) -> tuple[str, str | None]:
    """(the key with its page member as {page}, the page or None): pageVisibility.products.headerNavName →
    ("pageVisibility.{page}.headerNavName", "products")."""
    mm = PER_PAGE_KEY.match(str(key or ""))
    return (f"{mm.group(1)}.{{page}}.{mm.group(3)}", mm.group(2)) if mm else (str(key or ""), None)


def _norm_value(v: Any) -> str:
    t = re.sub(r"<[^>]+>", " ", str(v if v is not None else "")).strip().strip("'\"“”‘’").lower()
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"(?<=\d)px$", "", t)


def exam_answers(leak_index=None, exam_cases: list[dict] | None = None) -> list[dict]:
    """[{id, key, shape, page, values}] — every exam item's written key with the values its text states (leak.py
    literal_values) and, for a generated exam case (F0H), the values its expected ops write."""
    from .exam import AMENDMENTS, expected_sets, optional_keys
    from .leak import literal_values
    out: list[dict] = []
    for e in (getattr(leak_index, "entries", None) or []):
        vals = {_norm_value(v) for v in e.get("values") or set()}
        keys = {key for _, _, key in e.get("fkeys") or set()}
        amended = AMENDMENTS.get(tuple(str(e.get("id") or "").split(":", 1)))  # today's answer key (exam.amend)
        if amended:
            keys |= {str(x.get("key") or "") for s in expected_sets({"expected": amended}) + [optional_keys({"expected": amended})]
                     for x in s if isinstance(x, dict) and x.get("key")}
        for key in sorted(keys):
            shape, page = _key_shape(key)
            out.append({"id": e.get("id"), "key": key, "shape": shape, "page": page, "values": vals})
    for c in exam_cases or []:
        exp = c.get("expected") or {}
        vals = {_norm_value(v) for t in [c.get("request")] + [h.get("user") for h in c.get("history") or []]
                for v in literal_values(str(t or ""))}
        keys = set()
        for o in list(exp.get("ops") or []) + list((c.get("meta") or {}).get("oracle_ops") or []):
            for k, v in (o.get("values") or {}).items() if isinstance(o.get("values"), dict) else []:
                keys.add(k)
                vals.add(_norm_value(v))
        for s in [exp.get("keys") or []] + [a for a in exp.get("alt") or [] if isinstance(a, list)]:
            keys |= {str(x.get("key") or "") for x in s if isinstance(x, dict) and x.get("key")}
        for key in keys:
            shape, page = _key_shape(key)
            out.append({"id": str(c.get("id")), "key": key, "shape": shape, "page": page, "values": vals - {""}})
    return out


def exam_answer_clash(c: dict, answers: list[dict], sides: tuple[Side, ...] = SIDES + SITE_FONT_SIDES) -> str | None:
    """The exam item a case teaches the answer of (module docstring, leak), or None. The case's field and every
    look-alike (the sides of its pair) are compared by key shape; a hit needs the same value or the same page."""
    meta = (c.get("meta") or {}).get("contrast") or {}
    side = next((s for s in sides if s.name == meta.get("side")), None)
    if side is None or not answers:
        return None
    shapes = {s.key for s in sides if s.pair == side.pair} | {_key_shape(k)[0] for k in meta.get("keys") or []}
    value = _norm_value(meta.get("value")) if meta.get("value") is not None else None
    page = meta.get("page_key")
    for a in answers:
        if a["shape"] not in shapes and a["key"] not in shapes:
            continue
        if value and value in a["values"]:
            return f"exam_answer:{a['id']}:value"
        if page and a["page"] == page and a["shape"] in shapes and "{page}" in a["shape"]:
            return f"exam_answer:{a['id']}:page"
    return None


def validate(c: dict, m, engine, extras: dict, *, leak_index=None, seen: set[str] | None = None,
             answers: list[dict] | None = None) -> str | None:
    """None, or why the case is dropped (module docstring). Measures the case in place."""
    from .manifest import is_holdout
    from .r4cases import authored_values, dedup_key, remeasure, to_keys, write_ids
    probe = c
    if c["score"]["mode"] == "ask" and (c.get("meta") or {}).get("contrast"):  # the field a vague case names
        probe = {**c, "expected": {**c["expected"], "ops": [{"op": "set_fields", "target": c["target"],
                                                             "values": {c["meta"]["contrast"]["key"]: None}}]}}
    hk = sorted(k for k in write_ids(probe, m) if is_holdout(k))
    if hk:
        return "holdout_key:" + hk[0]
    clash = exam_answer_clash(c, answers or []) or page_exam_clash(c)
    if clash:
        return "leak:" + clash
    if c["score"]["mode"] == "exact":
        w = authored_values(c)
        if w:
            to_keys(c, w)
    err = remeasure(c, engine, m, extras)
    if err:
        return f"oracle:{err}"
    if isinstance(c["expected"].get("site_fonts"), dict):  # every site font, on the humanized state
        from .r4score import to_all_fonts
        c.update(to_all_fonts(c, m, c["expected"]["site_fonts"].get("value")))
        err = remeasure(c, engine, m, extras)
        if err:
            return f"oracle:{err}"
    from .r4score import closed_gate_writes
    from .verify import ctx_of
    ops = [o for o in c["meta"].get("oracle_ops") or c["expected"].get("ops") or [] if o.get("op") != "pick_asset"]
    if ops:
        from .createpage import run_engine_result
        r = run_engine_result(engine, copy.deepcopy(c["site_state"]), copy.deepcopy(ops), ctx_of(c))
        res = r.get("results") if isinstance(r, dict) else getattr(r, "results", [])
        if closed_gate_writes(res or []):
            return "hidden_write"  # a W_HIDDEN alone is not: the engine warns for every gated field, open or closed
    c["engine_fingerprint"] = getattr(engine, "fingerprint", None)
    if leak_index is not None:
        res = leak_index.check(c)
        if res["rule1"] or res["rule2"] or res["rule3"]:
            return "leak:rule" + ("1" if res["rule1"] else "2" if res["rule2"] else "3")
        if res["max_jaccard"] >= NEAR_EXAM:
            return "leak:near_exam"
        from .leak import CHECKER
        c["meta"]["leak"] = {"max_jaccard": res["max_jaccard"], "nearest": res["nearest"], "checker": CHECKER}
    if seen is not None:
        k = dedup_key(c)
        if k in seen:
            return "duplicate"
        seen.add(k)
    return None


def contrast_round(m, engine, extras: dict | None = None, *, leak_index=None, pool: list[dict] = (),
                   exam_cases: list[dict] | None = None, sides: tuple[Side, ...] = SIDES,
                   site_font_sides: tuple[Side, ...] = SITE_FONT_SIDES, vague: bool = True, ask_first: bool = True,
                   page_jobs: tuple = PAGE_JOBS, list_jobs: tuple = LIST_JOBS,
                   progress: Callable[[str], None] = print) -> tuple[list[dict], dict]:
    """(F8 cases + the ask-first cases, report). `pool`: cases already held (dedup; the source of ask_first_cases);
    `exam_cases`: the F0H cases, to report how many of them the family moves from key-unseen to key-seen
    (r4cases.tag_seen's split). After `sides`: `site_font_sides`, then one `vague` (mode ask) case per field of
    `sides` (module docstring), numbered on; then ask_first_cases from the pool."""
    from .r4cases import dedup_key, write_ids
    from .state import load_context_fixture
    extras = extras if extras is not None else load_context_fixture()
    problems = check_templates(m, tuple(sides) + tuple(site_font_sides))
    seen = {dedup_key(c) for c in pool or []}
    answers = exam_answers(leak_index, exam_cases)
    clashes: Counter = Counter()
    out: list[dict] = []
    dropped: Counter = Counter()
    per_side: dict[str, Counter] = {}
    tag = GENERATOR.replace("-", "")  # stable across manifests: carry-forward matches cases by id
    # (number, side, i, style, template): the pairs from 1, the site fonts from SITE_FONT_BASE, the vague cases from
    # VAGUE_BASE — so a group's ids never move when another grows
    jobs: list[tuple[int, Side, int, str, str]] = []
    for base, group in ((0, tuple(sides)), (SITE_FONT_BASE, tuple(site_font_sides))):
        n = base
        for s in group:
            styles = list(s.templates)
            for i in range(s.n):
                n += 1
                jobs.append((n, s, i, *styles[(i + h8(s.name)) % len(styles)]))
    if vague:
        n = VAGUE_BASE
        done_fields: set[tuple] = set()
        for s in sides:
            k = (s.kind, s.container, s.key)
            if k in done_fields:
                continue
            done_fields.add(k)
            n += 1
            jobs.append((n, s, 0, "vague", VAGUE_TEMPLATES[h8(VAGUE_GENERATOR + s.name) % len(VAGUE_TEMPLATES)]))
    for k, (group, request, spec) in enumerate(page_jobs or ()):
        jobs.append((PAGE_BASE + k + 1, None, k, "page:" + group, json.dumps([request, spec])))
    for k, job in enumerate(list_jobs or ()):
        jobs.append((LIST_BASE + k + 1, None, k, f"list:{job[1]}:{job[2]}", json.dumps(list(job))))
    for n, s, i, style, template in jobs:
        cid = f"{tag}-{FAMILY}-{n:04d}"
        if s is None:  # the capability groups
            name = style
            rec = per_side.setdefault(name, Counter())
            args = json.loads(template)
            case, why = (build_page_case(m, style.split(":", 1)[1], args[0], args[1], cid) if style.startswith("page:")
                         else build_list_case(m, *args[:4], cid, *args[4:]))
        else:
            name = f"vague:{s.name}" if style == "vague" else s.name
            rec = per_side.setdefault(name, Counter())
            case, why = build_vague_case(m, s, template, cid) if style == "vague" else build_case(m, s, i, style, template, cid)
        if case is not None:
            why = validate(case, m, engine, extras, leak_index=leak_index, seen=seen, answers=answers) or ""
        if why.startswith("leak:exam_answer"):
            clashes[why[len("leak:exam_answer:"):]] += 1
        if why:
            k = ("leak:exam_answer" if why.startswith("leak:exam_answer") else
                 why if why.startswith("holdout_key") else why.split(":")[0])
            dropped[k] += 1
            rec["dropped:" + ("exam_answer" if k == "leak:exam_answer" else why.split(":")[0])] += 1
            continue
        rec["kept"] += 1
        out.append(case)
    for name, rec in per_side.items():
        progress(f"contrast: {name}: {rec['kept']}/{rec['kept'] + sum(v for k, v in rec.items() if k != 'kept')} kept")
    f8 = list(out)
    ask_rep: dict[str, Any] = {"skipped": True}
    if ask_first and pool:
        af, ask_rep = ask_first_cases(list(pool), m, engine, extras, leak_index=leak_index, seen=seen)
        out += af
        progress(f"contrast: ask-first {len(af)} kept {ask_rep.get('dropped')}")
    report: dict[str, Any] = {
        "generator": GENERATOR, "family": FAMILY, "manifest_hash": m.hash, "asked": len(jobs),
        "kept": len(f8), "dropped": dict(dropped), "template_problems": problems,
        "exam_answer_clashes": dict(clashes),
        "per_side": {k: dict(v) for k, v in per_side.items()},
        "pairs": dict(Counter(c["meta"]["contrast"]["pair"] for c in f8)),
        "modes": dict(Counter(c["score"]["mode"] for c in f8)),
        "styles": dict(Counter(c["meta"]["style"] for c in f8)),
        "keys": dict(Counter(c["meta"]["contrast"]["key"] for c in f8)),
        "site_fonts": sum(1 for c in f8 if c["meta"]["contrast"]["pair"] == "site_fonts"),
        "vague": sum(1 for c in f8 if c["meta"]["contrast"]["pair"] == "vague"),
        "capabilities": capability_counts(f8),
        "capability_dropped": {k: dict(v) for k, v in per_side.items() if k.startswith(("page:", "list:"))
                               and any(x.startswith("dropped:") for x in v)},
        "ask_first": ask_rep,
        "total": len(out),
    }
    if exam_cases:
        mine = set().union(*(write_ids(c, m) for c in f8)) if f8 else set()
        before = {str(c.get("id")) for c in exam_cases if (c.get("meta") or {}).get("key_seen")}
        moved = [str(c.get("id")) for c in exam_cases
                 if str(c.get("id")) not in before and (c.get("score") or {}).get("mode") not in ("where", "clarify")
                 and c.get("kind") not in ("unsupported", "clarify") and write_ids(c, m) & mine]
        report["exam_unseen_to_seen"] = {"count": len(moved), "ids": moved[:50]}
    return out, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="souschef_lab.contrast", description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--why", type=Path, help="the paraphrase study folder (real conversations for the hold-out check)")
    ap.add_argument("--pool", type=Path, help="cases already held (dedup; their F0H for the seen/unseen report)")
    ap.add_argument("--engine", type=Path, help="headless.mjs (default $ENGINE / $COOKIE_CUTTER_DIR)")
    a = ap.parse_args(argv)
    from .engine import PersistentEngine, default_cache_dir
    from .leak import ExamIndex
    from .manifest import load
    from .r4cases import read_jsonl, write_jsonl
    m = load(str(a.manifest))
    pool = read_jsonl(a.pool) if a.pool else []
    eng = PersistentEngine(a.engine, manifest_path=a.manifest, size=4, cache_dir=default_cache_dir())
    try:
        cases, rep = contrast_round(m, eng, leak_index=ExamIndex(why=a.why), pool=[c for c in pool if not c.get("holdout")],
                                    exam_cases=[c for c in pool if (c.get("meta") or {}).get("family") == "F0H"],
                                    progress=lambda s: print(s, file=sys.stderr))
    finally:
        eng.close()
    write_jsonl(a.out, cases)
    rp = a.report or a.out.with_name(a.out.stem + "-report.json")
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("asked", "kept", "dropped", "pairs", "modes", "styles", "template_problems")},
                     indent=1))
    return 0 if not rep["template_problems"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
