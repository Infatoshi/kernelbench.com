// Minimal top chrome: wordmark (not a link) + GitHub + X. No nav.

export function SiteBrand() {
  return (
    <header className="site-brand">
      <div className="site-brand-inner">
        <span className="nav-wordmark site-brand-mark" aria-label="kernelbench.com">
          <span className="nav-word">kernelbench</span>
          <span className="nav-tld">.com</span>
        </span>
        <div className="nav-icons">
          <a
            href="https://github.com/Infatoshi/kernelbench.com"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            className="nav-icon no-underline"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.22-3.37-1.22-.45-1.19-1.11-1.5-1.11-1.5-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.9 1.57 2.35 1.12 2.92.86.09-.67.35-1.12.63-1.38-2.22-.26-4.55-1.14-4.55-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05A9.29 9.29 0 0 1 12 6.98c.85 0 1.71.12 2.51.34 1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.06.36.32.68.94.68 1.9 0 1.38-.01 2.49-.01 2.83 0 .27.18.59.69.49A10.05 10.05 0 0 0 22 12.26C22 6.58 17.52 2 12 2Z" />
            </svg>
          </a>
          <a
            href="https://x.com/elliotarledge"
            target="_blank"
            rel="noreferrer"
            aria-label="@elliotarledge on X"
            className="nav-icon no-underline"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
              <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
            </svg>
          </a>
        </div>
      </div>
    </header>
  )
}
