"use client"

// Scraper-resistant contact link: the address never appears in the served
// HTML (harvesters read static markup; almost none execute JS). Assembled
// on the client from parts and only materialized into an href on hover /
// focus / tap. Swap CONTACT_FORM_URL in for the mailto once the Google Form
// exists -- the component then renders a plain form link and this trick
// retires.
import { useState } from "react"

const CONTACT_FORM_URL: string | null = null

const U = ["elliot"]
const D = ["arledge", "net"]

export function ContactLink() {
  const [href, setHref] = useState<string | null>(null)
  if (CONTACT_FORM_URL) {
    return <a href={CONTACT_FORM_URL}>contact</a>
  }
  const addr = `${U.join("")}@${D.join(".")}`
  const arm = () => setHref(`mailto:${addr}`)
  return (
    <a href={href ?? "#contact"} onMouseEnter={arm} onFocus={arm} onTouchStart={arm}>
      {addr.replace("@", " [at] ")}
    </a>
  )
}
