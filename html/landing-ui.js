// ------------------------------------------------------------
    // UI micro-interactions (v2) — no API logic touched
    // ------------------------------------------------------------
    (function(){
      const prefersReduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

      // Mouse‑parallax for hero orbs (lightweight)
      if (!prefersReduce){
        const hero = document.querySelector("header.hero");
        if (hero){
          let raf = 0;
          const onMove = (ev)=>{
            const r = hero.getBoundingClientRect();
            const nx = ((ev.clientX - r.left) / r.width) * 2 - 1; // -1..1
            const ny = ((ev.clientY - r.top) / r.height) * 2 - 1;
            if (raf) cancelAnimationFrame(raf);
            raf = requestAnimationFrame(()=>{
              document.documentElement.style.setProperty("--px", nx.toFixed(3));
              document.documentElement.style.setProperty("--py", ny.toFixed(3));
            });
          };
          hero.addEventListener("pointermove", onMove, { passive:true });
          hero.addEventListener("pointerleave", ()=>{
            document.documentElement.style.setProperty("--px", "0");
            document.documentElement.style.setProperty("--py", "0");
          }, { passive:true });
        }
      }


      // Reveal on scroll
      const els = Array.from(document.querySelectorAll('.reveal'));
      if (!prefersReduce && 'IntersectionObserver' in window){
        const io = new IntersectionObserver((entries)=>{
          for (const e of entries){
            if (e.isIntersecting){
              e.target.classList.add('in');
              io.unobserve(e.target);
            }
          }
        }, { threshold: 0.14, rootMargin: '80px' });
        els.forEach(el => io.observe(el));
      } else {
        els.forEach(el => el.classList.add('in'));
      }

      // Button glow tracking for primary CTAs
      const primaries = Array.from(document.querySelectorAll('.btn.primary'));
      
      // Haptic tap on buttons (all .btn)
      const allBtns = Array.from(document.querySelectorAll('.btn'));
      allBtns.forEach(b=>{
        b.addEventListener('click', ()=>{
          b.classList.remove('haptic');
          void b.offsetWidth; // reflow to restart animation
          b.classList.add('haptic');
          setTimeout(()=> b.classList.remove('haptic'), 260);
        }, { passive:true });
      });

primaries.forEach(btn=>{
        btn.addEventListener('pointermove', (ev)=>{
          const r = btn.getBoundingClientRect();
          const x = ((ev.clientX - r.left) / r.width) * 100;
          const y = ((ev.clientY - r.top) / r.height) * 100;
          btn.style.setProperty('--mx', x.toFixed(1) + '%');
          btn.style.setProperty('--my', y.toFixed(1) + '%');
        });
      });

      // Shatter/assemble headline (hero)
      const shatters = Array.from(document.querySelectorAll('[data-fx="shatter"]'));
      shatters.forEach(el=>{
        // Preserve <br> tags and keep line breaks between words, not between letters.
        const nodes = Array.from(el.childNodes);
        el.textContent = "";
        nodes.forEach(n=>{
          if (n.nodeType === Node.ELEMENT_NODE && n.nodeName === "BR"){
            el.appendChild(document.createElement("br"));
            return;
          }
          const text = (n.nodeType === Node.TEXT_NODE) ? n.textContent : (n.textContent || "");
          const tokens = text.split(/(\s+)/);
          tokens.forEach(token=>{
            if (!token) return;
            if (/^\s+$/.test(token)){
              el.appendChild(document.createTextNode(token));
              return;
            }
            const word = document.createElement("span");
            word.className = "fx-word";
            for (const ch of token){
              const s = document.createElement("span");
              s.className = "fx-char";
              s.textContent = ch;
              // random offsets (small, premium — not glitchy)
              const dx = (Math.random()*34 - 17).toFixed(1);
              const dy = (Math.random()*26 - 13).toFixed(1);
              const dr = (Math.random()*22 - 11).toFixed(1);
              const dd = Math.floor(Math.random()*260);
              s.style.setProperty("--dx", dx + "px");
              s.style.setProperty("--dy", dy + "px");
              s.style.setProperty("--dr", dr + "deg");
              s.style.setProperty("--dd", dd + "ms");
              word.appendChild(s);
            }
            el.appendChild(word);
          });
        });
        // Run on next frame
        requestAnimationFrame(()=> el.classList.add("run"));
      });

      // Particle bursts on primary buttons (subtle)
      const burst = (btn, ev)=>{
        if (prefersReduce) return;
        const rect = btn.getBoundingClientRect();
        const x = (ev?.clientX ?? (rect.left + rect.width/2)) - rect.left;
        const y = (ev?.clientY ?? (rect.top + rect.height/2)) - rect.top;

        // Ensure wrapper
        let wrap = btn.querySelector(".burst-wrap");
        if (!wrap){
          wrap = document.createElement("div");
          wrap.className = "burst-wrap";
          btn.style.position = btn.style.position || "relative";
          btn.appendChild(wrap);
        }
        // Create particles
        const n = 12;
        for (let i=0;i<n;i++){
          const p = document.createElement("div");
          p.className = "p" + (i%4===0 ? " alt" : "");
          p.style.left = x + "px";
          p.style.top  = y + "px";
          const ang = Math.random() * Math.PI * 2;
          const mag = 18 + Math.random()*24;
          p.style.setProperty("--px", (Math.cos(ang)*mag).toFixed(1) + "px");
          p.style.setProperty("--py", (Math.sin(ang)*mag).toFixed(1) + "px");
          wrap.appendChild(p);
          p.addEventListener("animationend", ()=> p.remove(), { once:true });
        }
      };

      primaries.forEach(btn=>{
        // burst on click and on first hover (throttled)
        btn.addEventListener("click", (ev)=> burst(btn, ev));
        let armed = true;
        btn.addEventListener("pointerenter", (ev)=>{
          if (!armed) return;
          armed = false;
          burst(btn, ev);
          setTimeout(()=>{ armed = true; }, 1200);
        });
      });

    })();
