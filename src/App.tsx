import { useEffect, useRef, useState } from 'react'
import { ChalkDust } from './chalk/ChalkDust'
import { createChalkGui } from './chalk/gui'
import './App.css'

// Serializes WebGPU setup/teardown so StrictMode's double mount never races two surfaces on one canvas.
let lifecycle = Promise.resolve()

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const heroRef = useRef<HTMLElement>(null)
  const headlineRef = useRef<HTMLHeadingElement>(null)
  const productRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const dust = new ChalkDust(canvas)
    let cancelled = false
    let cleanup = () => {}

    lifecycle = lifecycle.then(async () => {
      if (cancelled) return
      try {
        await dust.start()
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
        return
      }
      if (cancelled) return

      dust.anchorFloor(heroRef.current) // the floor belongs to the hero and scrolls away with it
      const burstHeadline = () => headlineRef.current && dust.burstAtElement(headlineRef.current, 1.2)
      const gui = createChalkGui(dust, burstHeadline)
      setTimeout(burstHeadline, 300) // intro puff behind the headline

      // Scroll-triggered burst behind the product, once each time it enters view.
      const observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting) dust.burstAtElement(entry.target, 1)
          }
        },
        { threshold: 0.6 },
      )
      if (productRef.current) observer.observe(productRef.current)

      cleanup = () => {
        observer.disconnect()
        gui.destroy()
      }
    })

    return () => {
      cancelled = true
      lifecycle = lifecycle.then(() => {
        cleanup()
        dust.dispose()
      })
    }
  }, [])

  return (
    <>
      <canvas ref={canvasRef} className="chalk-canvas" />

      <header className="site-header">
        <span className="logo">ACME Shop</span>
        <span className="cart">Warenkorb 0</span>
      </header>

      <main>
        <section ref={heroRef} className="hero">
          <h1 ref={headlineRef}>Entwickelt für Dich.</h1>
          <a className="cta" href="#produkt">
            Jetzt shoppen
          </a>
          {error && <p className="gpu-error">WebGPU nicht verfügbar: {error}</p>}
        </section>

        <section className="product" id="produkt">
          <div ref={productRef} className="product-card">
            <span className="eyebrow">Neu</span>
            <h2>Olympic Barbell 20 kg</h2>
            <p>Scroll hierher und der Kreidestaub explodiert hinter dem Produkt.</p>
          </div>
        </section>

        <section className="spacer" />
      </main>
    </>
  )
}

export default App
