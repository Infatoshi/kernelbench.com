"use client"

import { useEffect, useState } from "react"

type Theme = "dark" | "light"

function currentTheme(): Theme {
  if (typeof document === "undefined") return "dark"
  return document.documentElement.dataset.theme === "light" ? "light" : "dark"
}

function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem("kb-theme", theme)
  } catch {
    // Ignore storage failures; the button should still update this page.
  }
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark")

  useEffect(() => {
    setTheme(currentTheme())
  }, [])

  const nextTheme: Theme = theme === "dark" ? "light" : "dark"

  return (
    <button
      type="button"
      className="theme-toggle"
      aria-label={`Switch to ${nextTheme} mode`}
      aria-pressed={theme === "light"}
      onClick={() => {
        applyTheme(nextTheme)
        setTheme(nextTheme)
      }}
    >
      {theme === "dark" ? "Light" : "Dark"}
    </button>
  )
}
