// Document Concierge — shared design tokens
//
// Single source of truth for the Tailwind config, included on every page.
// Token *names* are unchanged from the previous theme (nothing in app.js
// needed to change) — only the values were redrawn around an archive /
// card-catalogue direction instead of the previous violet glass look.

tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Surfaces — parchment, not violet-tinted white
        "background": "#EDEAE0",
        "surface": "#EDEAE0",
        "surface-bright": "#F7F5EE",
        "surface-dim": "#DED9C9",
        "surface-container-lowest": "#F7F5EE",
        "surface-container-low": "#F1EDE1",
        "surface-container": "#EAE5D6",
        "surface-container-high": "#E3DDC9",
        "surface-container-highest": "#DCD5BD",
        "surface-variant": "#E3DDC9",
        "surface-tint": "#2F4B3C",

        // Text / outline — warm ink instead of violet-black
        "on-background": "#201E19",
        "on-surface": "#201E19",
        "on-surface-variant": "#4A463C",
        "text-muted": "#6B6558",
        "outline": "#8A8474",
        "outline-variant": "#D8D3C4",
        "glass-border": "rgba(32, 30, 25, 0.14)",
        "accent-glow": "rgba(47, 75, 60, 0.16)",
        "inverse-surface": "#332F27",
        "inverse-on-surface": "#F3EFE3",
        "inverse-primary": "#9CC2AC",

        // Primary — deep ledger green (replaces violet)
        "primary": "#2F4B3C",
        "primary-container": "#3F614E",
        "primary-fixed": "#D9E4DC",
        "primary-fixed-dim": "#9CC2AC",
        "on-primary": "#F7F5EE",
        "on-primary-container": "#EAF1EC",
        "on-primary-fixed": "#13251C",
        "on-primary-fixed-variant": "#1D2F26",

        // Secondary — muted clay, used sparingly (e.g. one filetype accent)
        "secondary": "#8B4A3C",
        "secondary-container": "#C97C63",
        "secondary-fixed": "#F1DCD3",
        "secondary-fixed-dim": "#D9A793",
        "on-secondary": "#FFFFFF",
        "on-secondary-container": "#3B1F17",
        "on-secondary-fixed": "#3B1F17",
        "on-secondary-fixed-variant": "#63382B",

        // Tertiary — the highlighter amber, reserved for citation/annotation-style accents
        "tertiary": "#8A6415",
        "tertiary-container": "#E8B23D",
        "tertiary-fixed": "#F6E3B8",
        "tertiary-fixed-dim": "#E8B23D",
        "on-tertiary": "#FFFFFF",
        "on-tertiary-container": "#3A2A06",
        "on-tertiary-fixed": "#2A1E04",
        "on-tertiary-fixed-variant": "#5C4310",

        "error": "#A6342A",
        "error-container": "#F3D9D3",
        "on-error": "#FFFFFF",
        "on-error-container": "#4A140D",

        "bg-cream": "#F7F5EE",
        "bg-warm": "#F7F5EE",

        // Token-budget bar segments (cache/ask diagnostics)
        "router-seg": "#2F4B3C",
        "fanout-seg": "#3D5A6B",
        "agg-seg": "#E8B23D",
      },
      borderRadius: {
        DEFAULT: "0.125rem",
        lg: "0.1875rem",
        xl: "0.25rem",
        full: "9999px",
      },
      spacing: {
        "gap-md": "16px",
        "gap-lg": "24px",
        "gap-sm": "12px",
        "gap-xs": "8px",
        "container-max": "880px",
        base: "4px",
        "page-margin": "28px",
      },
      fontFamily: {
        "body-lg": ["IBM Plex Sans", "sans-serif"],
        "body-sm": ["IBM Plex Sans", "sans-serif"],
        "headline-md": ["Fraunces", "serif"],
        "headline-lg": ["Fraunces", "serif"],
        "data-num": ["IBM Plex Mono", "monospace"],
        "label-mono-bold": ["IBM Plex Mono", "monospace"],
        "micro-label": ["IBM Plex Mono", "monospace"],
      },
      fontSize: {
        "body-lg": ["15px", { lineHeight: "24px", fontWeight: "400" }],
        "data-num": ["12px", { lineHeight: "16px", fontWeight: "500" }],
        "headline-md": ["18px", { lineHeight: "24px", fontWeight: "600" }],
        "headline-lg": ["24px", { lineHeight: "32px", letterSpacing: "-0.01em", fontWeight: "700" }],
        "label-mono-bold": ["11px", { lineHeight: "16px", letterSpacing: "0.14em", fontWeight: "600" }],
        "micro-label": ["9px", { lineHeight: "12px", fontWeight: "500" }],
        "body-sm": ["13px", { lineHeight: "20px", fontWeight: "400" }],
      },
    },
  },
};
