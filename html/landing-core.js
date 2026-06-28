
    // ------------------------------------------------------------
    // Demo routing
    // IMPORTANT: adjust the demoPath to your real route/path.
    // If you keep both HTML files in one folder, this works.
    // ------------------------------------------------------------
    const demoPath = "/dashboard";
    const PAGE_PARAMS = new URLSearchParams(window.location.search);
    const SHOULD_FORCE_LOGIN = PAGE_PARAMS.get("login") === "1" || PAGE_PARAMS.get("auth") === "login";
    const NEXT_AFTER_LOGIN = (() => {
      const raw = (PAGE_PARAMS.get("next") || "").trim();
      if (!raw.startsWith("/") || raw.startsWith("//")) return "";
      return raw;
    })();
    const OAUTH_PROVIDER = (PAGE_PARAMS.get("oauth") || "").trim().toLowerCase();
    const OAUTH_ERROR = (PAGE_PARAMS.get("oauth_error") || "").trim();

    // ------------------------------------------------------------
    // Minimal auth + backend registration/login
    // ------------------------------------------------------------
    const LS_AUTH = "tf_auth_v1";  // JSON: { email, token }
    const LS_SUB  = "tf_sub_v1";   // "active" | "none"
    const SUPPORT_EMAIL = "support@tradeforge.art";
    const SUPPORT_TG = "https://t.me/TRADE_FORGE333";
    const BILLING_PERIODS = {
      monthly: { id: "monthly", label: "1 месяц", suffix: "/ месяц" },
      quarterly: { id: "quarterly", label: "3 месяца", suffix: "/ 3 месяца" },
      semiannual: { id: "semiannual", label: "6 месяцев", suffix: "/ 6 месяцев" },
      yearly: { id: "yearly", label: "12 месяцев", suffix: "/ 12 месяцев" },
    };
    const PLANS = {
      starter: { id: "starter", name: "Starter", prices: { monthly: 3333, quarterly: 8990, semiannual: 16990, yearly: 29990 } },
      pro: { id: "pro", name: "Pro", prices: { monthly: 4890, quarterly: 13290, semiannual: 24990, yearly: 43990 } },
      elite: { id: "elite", name: "Elite", prices: { monthly: 7890, quarterly: 21490, semiannual: 39990, yearly: 69990 } },
    };
    let selectedBillingPeriod = BILLING_PERIODS.monthly.id;
    let selectedPlan = PLANS.pro;
    let wantPaywall = false; // v6: show paywall after email verification

    const qs = (s, root=document) => root.querySelector(s);

    const API_BASES = { v6: "https://api.tradeforge.art" };
    const configReady = Promise.resolve();
    const apiFetch = (url, opts = {}) => {
      const headers = Object.assign({}, opts.headers || {});
      const token = getAuth()?.token;
      if (token) headers["Authorization"] = `Bearer ${token}`;
      return fetch(url, Object.assign({}, opts, { headers }));
    };

    const toastEl = qs("#toast");
    let toastTimer = null;
    function toast(msg){
      toastEl.textContent = msg;
      toastEl.classList.add("show");
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => toastEl.classList.remove("show"), 1400);
    }

    function getAuth(){
      try { return JSON.parse(localStorage.getItem(LS_AUTH) || "null"); } catch(e){ return null; }
    }
    function setAuth(auth){
      localStorage.setItem(LS_AUTH, JSON.stringify(auth));
      syncUI();
    }
    function clearAuth(){
      localStorage.removeItem(LS_AUTH);
      syncUI();
    }
    function getSub(){
      return localStorage.getItem(LS_SUB) || "none";
    }
    function setSub(status){
      localStorage.setItem(LS_SUB, status);
      syncUI();
    }

    function isAuthed(){ return !!getAuth()?.email; }
    function hasActiveSub(){ return getSub() === "active"; }
    function clearOauthQuery(){
      try{
        const url = new URL(window.location.href);
        url.searchParams.delete("oauth");
        url.searchParams.delete("oauth_error");
        history.replaceState({}, "", url.pathname + (url.search ? url.search : "") + url.hash);
      }catch(_){}
    }
    function startYandexLogin(){
      const next = NEXT_AFTER_LOGIN || "";
      const url = `${API_BASES.v6}/auth/yandex/start${next ? `?next=${encodeURIComponent(next)}` : ""}`;
      window.location.href = url;
    }
    async function completeOauthBridge(){
      if (!OAUTH_PROVIDER && !OAUTH_ERROR) return false;
      if (OAUTH_ERROR){
        toast(OAUTH_ERROR);
        openModal("login");
        clearOauthQuery();
        return false;
      }
      try{
        await configReady;
        const resp = await fetch(`${API_BASES.v6}/auth/session-bridge`, {
          method: "GET",
          credentials: "include",
          cache: "no-store",
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data?.ok || !data?.email) throw new Error(data?.detail || "Не удалось завершить вход");
        setAuth({ email: data.email, token: data.token || "" });
        if (data.subscription) {
          setSub(data.subscription.active ? "active" : "none");
        } else {
          await refreshSubscription();
        }
        const next = NEXT_AFTER_LOGIN || "";
        clearOauthQuery();
        toast("Вход выполнен");
        if (next) {
          window.location.href = next;
          return true;
        }
        return true;
      }catch(e){
        toast(String(e?.message || e || "Ошибка social login"));
        openModal("login");
        clearOauthQuery();
        return false;
      }
    }
    async function refreshSubscription(){
      try{
        await configReady;
        const r = await apiFetch(`${API_BASES.v6}/subscription/status`);
        const data = await r.json().catch(()=>({}));
        if (r.ok && data && data.subscription){
          setSub(data.subscription.active ? "active" : "none");
        }
      }catch(e){}
    }

    // UI elements
    const overlay = qs("#overlay");
    const closeModal = qs("#closeModal");
    const authTabs = qs("#authTabs");
    const signupForm = qs("#signupForm");
    const loginForm = qs("#loginForm");
    const accountPanel = qs("#accountPanel");
    const suErr = qs("#suErr");
    const liErr = qs("#liErr");
    const suVerifyWrap = qs("#suVerifyWrap");
    const liVerifyWrap = qs("#liVerifyWrap");
    const liResetWrap = qs("#liResetWrap");
    const paywallBlock = qs("#paywallBlock");
    const paywallLogin = qs("#paywallLogin");
    const paywallPlanText = qs("#paywallPlanText");
    const paywallLoginText = qs("#paywallLoginText");

    const pillText = qs("#pillText");
    const statusPill = qs("#statusPill");
    const accountNodes = {
      email: qs("#accEmail"),
      plan: qs("#accPlan"),
      status: qs("#accStatus"),
      expires: qs("#accExpires"),
      updated: qs("#accUpdated"),
      fullName: qs("#accFullName"),
      telegram: qs("#accTelegram"),
      selectedPlan: qs("#accSelectedPlan"),
      hint: qs("#accHint"),
      userId: qs("#accUserId"),
      supportCode: qs("#accSupportCode"),
      paymentSummary: qs("#accPaymentSummary"),
      txid: qs("#accTxid"),
    };
    const accountCopySupportBtn = qs("#accountCopySupportBtn");
    const yandexButtons = [qs("#signupYandexBtn"), qs("#loginYandexBtn")].filter(Boolean);

    
    // Pricing highlight slider (v333)
    function ensurePricingHighlight(){
      const wrap = qs(".pricing");
      if (!wrap) return null;
      let hl = wrap.querySelector(".price-highlight");
      if (!hl){
        hl = document.createElement("div");
        hl.className = "price-highlight";
        wrap.prepend(hl);
        wrap.classList.add("has-highlight");
      }
      return hl;
    }
    function movePricingHighlight(card){
      const wrap = qs(".pricing");
      const hl = ensurePricingHighlight();
      if (!wrap || !hl || !card) return;
      const rWrap = wrap.getBoundingClientRect();
      const r = card.getBoundingClientRect();
      const x = Math.round(r.left - rWrap.left);
      const y = Math.round(r.top - rWrap.top);
      hl.style.setProperty("--hx", x + "px");
      hl.style.setProperty("--hy", y + "px");
      hl.style.width = Math.round(r.width) + "px";
      hl.style.height = Math.round(r.height) + "px";
      hl.style.opacity = "1";
    }

    function updatePaywallCopy(){
      const period = BILLING_PERIODS[selectedBillingPeriod] || BILLING_PERIODS.monthly;
      const amount = selectedPlan.prices[selectedBillingPeriod];
      if (paywallPlanText){
        paywallPlanText.innerHTML = `План <b>TradeForge ${selectedPlan.name}</b>: <b>3 дня trial</b>, затем <b>${amount} ₽ ${period.suffix}</b>. Оплата в рублях через Prodamus / СБП.`;
      }
      if (paywallLoginText){
        paywallLoginText.innerHTML = `Чтобы открыть демо, активируй trial 3 дня, затем <b>${amount} ₽ ${period.suffix}</b> через Prodamus / СБП.`;
      }
      if (accountNodes.selectedPlan){
        accountNodes.selectedPlan.textContent = `TradeForge ${selectedPlan.name} • ${amount} ₽ ${period.suffix}`;
      }
    }

    function updatePricingPeriodUI(){
      const period = BILLING_PERIODS[selectedBillingPeriod] || BILLING_PERIODS.monthly;
      Object.values(PLANS).forEach((plan) => {
        const priceEl = document.querySelector(`[data-price-for="${plan.id}"]`);
        const labelEl = document.querySelector(`[data-price-label-for="${plan.id}"]`);
        const amount = plan.prices[selectedBillingPeriod];
        if (priceEl) priceEl.textContent = `${amount} ₽`;
        if (labelEl) labelEl.textContent = period.suffix;
      });
      document.querySelectorAll("[data-billing-period]").forEach((btn) => {
        btn.classList.toggle("active", btn.getAttribute("data-billing-period") === selectedBillingPeriod);
      });
    }

    function setBillingPeriod(periodId){
      if (!BILLING_PERIODS[periodId]) return;
      selectedBillingPeriod = periodId;
      updatePricingPeriodUI();
      updatePaywallCopy();
    }

function setSelectedPlan(planId){
      selectedPlan = PLANS[planId] || PLANS.pro;
      updatePaywallCopy();
      updatePricingPeriodUI();

      // Pricing selection highlight (UI only)
      const cards = Array.from(document.querySelectorAll(".price-card"));
      cards.forEach(c => c.classList.remove("selected"));

      const btn = document.querySelector(`[data-plan-btn="${planId}"]`);
      const card = btn ? btn.closest(".price-card") : null;
      if (card) card.classList.add("selected");

      // ARIA pressed for accessibility
      const planBtns = Array.from(document.querySelectorAll("[data-plan-btn]"));
      planBtns.forEach(b => b.setAttribute("aria-pressed", "false"));
      if (btn) btn.setAttribute("aria-pressed", "true");

      const accountPlanBtns = Array.from(document.querySelectorAll("[data-account-plan]"));
      accountPlanBtns.forEach(b => {
        const active = b.getAttribute("data-account-plan") === selectedPlan.id;
        b.classList.toggle("primary", active);
        b.classList.toggle("ghost", false);
      });
    }

    function openModal(mode = "signup", { showPaywall = false } = {}){
      overlay.classList.add("show");
      document.body.style.overflow = "hidden";

      setTab(mode);
      if (mode === "account") {
        loadAccountPanel();
      }

      if (mode === "signup"){
        wantPaywall = !!showPaywall;
        // v6: keep paywall hidden until email is verified (prevents confusion)
        paywallBlock.style.display = "none";
      paywallBlock.classList.remove("reveal");
      const badge = qs("#suVerifiedBadge");
      if (badge){ badge.style.display = "none"; badge.classList.remove("show-check"); }
      } else {
        paywallLogin.style.display = "none";
      wantPaywall = false;
      }
    }

    function close(){
      overlay.classList.remove("show");
      document.body.style.overflow = "";
      suErr.classList.remove("show");
      liErr.classList.remove("show");
      suErr.textContent = "";
      liErr.textContent = "";
      suVerifyWrap.style.display = "none";
      liVerifyWrap.style.display = "none";
      liResetWrap.style.display = "none";
      paywallBlock.style.display = "none";
      paywallBlock.classList.remove("reveal");
      const badge = qs("#suVerifiedBadge");
      if (badge){ badge.style.display = "none"; badge.classList.remove("show-check"); }
      paywallLogin.style.display = "none";
      wantPaywall = false;
    }

    function setTab(tab){
      authTabs.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === tab));
      const isSignup = tab === "signup";
      const isAccount = tab === "account";
      signupForm.style.display = isSignup ? "flex" : "none";
      loginForm.style.display  = (!isSignup && !isAccount) ? "flex" : "none";
      if (accountPanel) accountPanel.style.display = isAccount ? "flex" : "none";
      qs("#modalTitle").textContent = isSignup ? "Create account" : (isAccount ? "Аккаунт" : "Sign in");
    }

    function formatUnixTs(ts){
      const num = Number(ts || 0);
      if (!Number.isFinite(num) || num <= 0) return "—";
      try{
        return new Date(num * 1000).toLocaleString("ru-RU");
      }catch(_){
        return "—";
      }
    }

    function formatPlanLabel(plan){
      const raw = String(plan || "").trim();
      if (!raw) return "—";
      if (raw === "owner") return "Owner";
      if (raw === "starter") return "Starter";
      if (raw === "pro") return "Pro";
      if (raw === "elite") return "Elite";
      return raw;
    }

    function formatRub(v){
      const num = Number(v || 0);
      if (!Number.isFinite(num) || num <= 0) return "—";
      try{
        return `${num.toLocaleString("ru-RU")} ₽`;
      }catch(_){
        return `${num} ₽`;
      }
    }

    async function loadAccountPanel(){
      const auth = getAuth();
      if (!auth?.email) {
        setTab("login");
        return;
      }
      if (accountNodes.email) accountNodes.email.textContent = auth.email;
      if (accountNodes.plan) accountNodes.plan.textContent = "loading...";
      if (accountNodes.status) accountNodes.status.textContent = "checking...";
      if (accountNodes.expires) accountNodes.expires.textContent = "—";
      if (accountNodes.updated) accountNodes.updated.textContent = "—";
      if (accountNodes.userId) accountNodes.userId.textContent = "—";
      if (accountNodes.supportCode) accountNodes.supportCode.textContent = "—";
      if (accountNodes.paymentSummary) accountNodes.paymentSummary.textContent = "—";
      if (accountNodes.txid) accountNodes.txid.textContent = "—";
      if (accountNodes.fullName) accountNodes.fullName.value = "";
      if (accountNodes.telegram) accountNodes.telegram.value = "";
      if (accountNodes.hint) accountNodes.hint.textContent = "Проверяем текущий статус подписки...";
      try{
        await configReady;
        const resp = await apiFetch(`${API_BASES.v6}/auth/me`, { credentials: "include", cache: "no-store" });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data?.subscription) {
          throw new Error(data?.detail || "Не удалось загрузить аккаунт");
        }
        const sub = data.subscription || {};
        setSub(sub.active ? "active" : "none");
        if (accountNodes.email) accountNodes.email.textContent = data.email || auth.email;
        if (accountNodes.plan) accountNodes.plan.textContent = formatPlanLabel(sub.plan);
        if (accountNodes.status) accountNodes.status.textContent = sub.active ? "Активна" : "Не активна";
        if (accountNodes.expires) accountNodes.expires.textContent = formatUnixTs(sub.expires_at);
        if (accountNodes.fullName) accountNodes.fullName.value = String(data?.profile?.full_name || "");
        if (accountNodes.telegram) accountNodes.telegram.value = String(data?.profile?.telegram || "");
        if (accountNodes.userId) accountNodes.userId.textContent = String(data?.support?.user_id || "—");
        if (accountNodes.supportCode) accountNodes.supportCode.textContent = String(data?.support?.support_code || "—");
        if (accountNodes.txid) {
          const tx = String(data?.support?.subscription_txid || data?.support?.latest_payment?.txid || "").trim();
          accountNodes.txid.textContent = tx || "—";
        }
        if (accountNodes.paymentSummary) {
          const p = data?.support?.latest_payment || null;
          accountNodes.paymentSummary.textContent = p
            ? `${p.provider || "payment"} • ${p.status || "—"} • ${formatRub(p.amount_rub)} • ${p.billing_period || "—"}`
            : "Платежей пока нет";
        }
        if (accountNodes.updated) {
          const ts = Number(data?.profile?.updated_at || 0) || Number(sub.updated_at || 0);
          accountNodes.updated.textContent = formatUnixTs(ts);
        }
        if (accountNodes.hint) {
          accountNodes.hint.textContent = sub.active
            ? "Подписка активна. Можно сразу открывать демо и рабочие модули."
            : "Подписка не активна. Можно оплатить доступ через Prodamus ниже.";
        }
        syncUI();
      }catch(e){
        if (accountNodes.plan) accountNodes.plan.textContent = "—";
        if (accountNodes.status) accountNodes.status.textContent = "Ошибка";
        if (accountNodes.hint) accountNodes.hint.textContent = String(e?.message || e || "Не удалось загрузить аккаунт");
      }
    }

    async function saveAccountProfile(){
      const auth = getAuth();
      if (!auth?.email) {
        setTab("login");
        return;
      }
      const full_name = String(accountNodes.fullName?.value || "").trim();
      const telegram = String(accountNodes.telegram?.value || "").trim();
      try{
        await configReady;
        const resp = await apiFetch(`${API_BASES.v6}/auth/profile`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ full_name, telegram }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data?.ok) {
          throw new Error(data?.detail || "Не удалось сохранить профиль");
        }
        if (accountNodes.updated) {
          accountNodes.updated.textContent = formatUnixTs(data?.profile?.updated_at);
        }
        if (accountNodes.hint) {
          accountNodes.hint.textContent = "Профиль сохранён. Эти данные можно обновлять в любой момент.";
        }
        toast("Профиль сохранён");
      }catch(e){
        toast(String(e?.message || e || "Ошибка сохранения профиля"));
      }
    }

    function buildSupportPayload(){
      return [
        `Email: ${accountNodes.email?.textContent || "—"}`,
        `Client ID: ${accountNodes.userId?.textContent || "—"}`,
        `Support code: ${accountNodes.supportCode?.textContent || "—"}`,
        `Plan: ${accountNodes.plan?.textContent || "—"}`,
        `Subscription status: ${accountNodes.status?.textContent || "—"}`,
        `Active until: ${accountNodes.expires?.textContent || "—"}`,
        `Payment: ${accountNodes.paymentSummary?.textContent || "—"}`,
        `TXID/Order: ${accountNodes.txid?.textContent || "—"}`,
        `Full name: ${accountNodes.fullName?.value?.trim() || "—"}`,
        `Telegram: ${accountNodes.telegram?.value?.trim() || "—"}`,
      ].join("\n");
    }

    async function copySupportPayload(){
      const text = buildSupportPayload();
      try{
        if (navigator.clipboard?.writeText){
          await navigator.clipboard.writeText(text);
        } else {
          throw new Error("clipboard unavailable");
        }
        toast("Данные для поддержки скопированы");
      }catch(_){
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        try { document.execCommand("copy"); } catch(__) {}
        document.body.removeChild(ta);
        toast("Данные выделены/скопированы");
      }
    }

    // Close handlers
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    closeModal.addEventListener("click", close);
    window.addEventListener("keydown", (e) => { if (e.key === "Escape" && overlay.classList.contains("show")) close(); });

    authTabs.addEventListener("click", (e) => {
      const t = e.target.closest(".tab");
      if (!t) return;
      setTab(t.dataset.tab);
    });
    qs("#goLogin").addEventListener("click", () => setTab("login"));
    qs("#goSignup").addEventListener("click", () => setTab("signup"));
    yandexButtons.forEach((btn) => btn.addEventListener("click", () => startYandexLogin()));

    // Links
    const scrollToLegal = () => {
      const legal = document.querySelector("#legal");
      if (legal) {
        legal.scrollIntoView({ behavior: "smooth", block: "start" });
      } else {
        toast("Открой блок с условиями и реквизитами ниже на странице.");
      }
    };
    qs("#termsLink").addEventListener("click", (e) => {
      e.preventDefault();
      scrollToLegal();
    });
    qs("#privacyLink").addEventListener("click", (e) => {
      e.preventDefault();
      scrollToLegal();
    });
    qs("#termsBtnInline").addEventListener("click", scrollToLegal);
    qs("#privacyBtnInline").addEventListener("click", scrollToLegal);
    qs("#forgotBtn").addEventListener("click", async () => {
      clearErr(liErr);
      const email = qs("#liEmail").value.trim();
      if (!email) return showErr(liErr, "Укажи email для восстановления.");
      try{
        await configReady;
        await apiFetch(`${API_BASES.v6}/auth/request-password-reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email })
        });
        liResetWrap.style.display = "block";
        qs("#liResetCode").focus();
        toast("Код сброса отправлен на email");
      }catch(e){
        showErr(liErr, "Сервис недоступен");
      }
    });

    // Top buttons
    qs("#loginBtnTop").addEventListener("click", () => {
      if (isAuthed()) return openModal("account");
      openModal("login");
    });
    qs("#ctaBtnTop").addEventListener("click", () => { setSelectedPlan("pro"); openModal("signup", { showPaywall: true }); });
    qs("#ctaHero").addEventListener("click", () => { setSelectedPlan("pro"); openModal("signup", { showPaywall: true }); });
    qs("#ctaPricing").addEventListener("click", () => { setSelectedPlan("pro"); openModal("signup", { showPaywall: true }); });
    qs("#ctaSticky").addEventListener("click", () => { setSelectedPlan("pro"); openModal("signup", { showPaywall: true }); });
    // Trial section buttons (added in v2; no backend changes)
    qs("#ctaTrial")?.addEventListener("click", () => { setSelectedPlan("pro"); openModal("signup", { showPaywall: true }); });
    qs("#demoBtnTrial")?.addEventListener("click", async (e) => { e.preventDefault(); await requireDemoAccess(); });

    document.querySelectorAll("[data-plan-btn]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const planId = btn.getAttribute("data-plan-btn") || "pro";
        setSelectedPlan(planId);
        if (isAuthed()) return openModal("account");
        openModal("signup", { showPaywall: true });
      });
    });

    document.querySelectorAll("[data-billing-period]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const periodId = btn.getAttribute("data-billing-period") || BILLING_PERIODS.monthly.id;
        setBillingPeriod(periodId);
      });
    });

    document.querySelectorAll("[data-account-plan]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const planId = btn.getAttribute("data-account-plan") || "pro";
        setSelectedPlan(planId);
        if (accountNodes.hint) {
          accountNodes.hint.textContent = `К оплате выбран TradeForge ${PLANS[planId].name}. Ниже можно сразу перейти в Prodamus.`;
        }
      });
    });

    // Account button
    qs("#manageBtn").addEventListener("click", () => {
      if (!isAuthed()) return openModal("login");
      openModal("account");
    });

    // Protected links to demo
    const protectedEls = [
      qs("#demoBtnHero"),
      qs("#openDemoInline"),
      qs("#openDemoFoot"),
      qs("#demoLinkTop"),
    ].filter(Boolean);

    function goDemo(){
      window.location.href = demoPath;
    }

    async function requireDemoAccess(){
      // Rule: need auth + active subscription
      if (!isAuthed()){
        toast("Нужен логин");
        openModal("login");
        return;
      }
      if (!hasActiveSub()){
        const auth = getAuth();
        if (auth?.token){
          await refreshSubscription();
        } else {
          setSub("none");
        }
      }
      if (!hasActiveSub()){
        toast("Нужна подписка (тарифы от 3333 ₽)");
        // show paywall inside login form
        openModal("login");
        paywallLogin.style.display = "block";
        return;
      }
      goDemo();
    }

    protectedEls.forEach(el => el.addEventListener("click", async (e) => {
      e.preventDefault();
      await requireDemoAccess();
    }));

    // Signup validation + placeholder auth
    function showErr(el, msg){
      el.textContent = msg;
      el.classList.add("show");
    }
    function clearErr(el){
      el.textContent = "";
      el.classList.remove("show");
    }

    signupForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      clearErr(suErr);
      suVerifyWrap.style.display = "none";

      const email = qs("#suEmail").value.trim();
      const p1 = qs("#suPass").value;
      const p2 = qs("#suPass2").value;
      const terms = qs("#suTerms").checked;

      if (!email) return showErr(suErr, "Укажи email.");
      if (p1.length < 10) return showErr(suErr, "Пароль должен быть минимум 10 символов.");
      if (p1 !== p2) return showErr(suErr, "Пароли не совпадают.");
      if (!terms) return showErr(suErr, "Нужно принять Terms & Privacy.");

      try{
        await configReady;
        const resp = await apiFetch(`${API_BASES.v6}/auth/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password: p1 })
        });
        if (!resp.ok){
          const data = await resp.json().catch(()=>({}));
          return showErr(suErr, data.detail || "Ошибка регистрации");
        }
        const regData = await resp.json().catch(()=>({}));
        if (regData && regData.email_sent === false){
          return showErr(suErr, "Почтовый сервис не настроен на сервере (SMTP).");
        }
        suVerifyWrap.style.display = "block";
        qs("#suCode").focus();
        toast("Код отправлен на email");
      }catch(e){
        showErr(suErr, "Сервис недоступен");
      }
    });

    qs("#suVerifyBtn").addEventListener("click", async () => {
      clearErr(suErr);
      const email = qs("#suEmail").value.trim();
      const code = qs("#suCode").value.trim();
      if (!email || !code) return showErr(suErr, "Укажи email и код.");
      try{
        await configReady;
        const vr = await apiFetch(`${API_BASES.v6}/auth/verify-email`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, code })
        });
        if (!vr.ok){
          const data = await vr.json().catch(()=>({}));
          return showErr(suErr, data.detail || "Код неверный/просрочен");
        }
        const vData = await vr.json().catch(()=>({}));
        setAuth({ email, token: vData?.token || "" });
        if (vData && vData.subscription){
          setSub(vData.subscription.active ? "active" : "none");
        } else {
          await refreshSubscription();
        }
        suVerifyWrap.style.display = "none";
        toast("Email подтвержден. Trial 3 дня активирован");
        // v6: optionally reveal billing block after verification
        if (wantPaywall){
          // If already active, keep hidden
          const st = getSub();
          if (!st || st !== "active"){
            paywallBlock.style.display = "block";
            paywallBlock.classList.add("reveal");
            const badge = qs("#suVerifiedBadge");
            if (badge){ badge.style.display = "flex"; badge.classList.remove("show-check"); void badge.offsetWidth; badge.classList.add("show-check"); }
            // smooth scroll to paywall
            try{ paywallBlock.scrollIntoView({ behavior: "smooth", block: "nearest" }); }catch(_){ }
          }
        }
        close();
        setTimeout(() => goDemo(), 250);
      }catch(e){
        showErr(suErr, "Сервис недоступен");
      }
    });

    qs("#suResendBtn").addEventListener("click", async () => {
      clearErr(suErr);
      const email = qs("#suEmail").value.trim();
      if (!email) return showErr(suErr, "Укажи email.");
      try{
        await configReady;
        const rr = await apiFetch(`${API_BASES.v6}/auth/resend-code`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email })
        });
        const data = await rr.json().catch(()=>({}));
        if (!rr.ok) return showErr(suErr, data.detail || "Не удалось отправить код");
        toast("Код отправлен повторно");
      }catch(e){
        showErr(suErr, "Сервис недоступен");
      }
    });

    // Login placeholder
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      clearErr(liErr);
      liVerifyWrap.style.display = "none";
      liResetWrap.style.display = "none";

      const email = qs("#liEmail").value.trim();
      const pass = qs("#liPass").value;

      if (!email || !pass) return showErr(liErr, "Укажи email и пароль.");

      try{
        await configReady;
        const resp = await fetch(`${API_BASES.v6}/auth/web-login`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password: pass })
        });
        if (!resp.ok){
          const data = await resp.json().catch(()=>({}));
          if (resp.status === 403 && (data.detail || "").toLowerCase().includes("not verified")){
            const rr = await apiFetch(`${API_BASES.v6}/auth/resend-code`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ email })
            });
            if (rr.ok){
              liVerifyWrap.style.display = "block";
              qs("#liCode").focus();
              return showErr(liErr, "Email не подтвержден. Новый код отправлен.");
            }
          }
          return showErr(liErr, data.detail || "Ошибка входа");
        }
        const loginData = await resp.json().catch(()=>({}));
        setAuth({ email, token: loginData?.token || "" });
        if (loginData && loginData.subscription){
          setSub(loginData.subscription.active ? "active" : "none");
        } else {
          await refreshSubscription();
        }
        toast("Вход выполнен");
      }catch(e){
        return showErr(liErr, "Сервис недоступен");
      }

      if (!hasActiveSub()){
        paywallLogin.style.display = "block";
      } else {
        close();
        toast("Доступ активен");
        if (NEXT_AFTER_LOGIN) {
          window.location.href = NEXT_AFTER_LOGIN;
        }
      }
    });

    qs("#liVerifyBtn").addEventListener("click", async () => {
      clearErr(liErr);
      const email = qs("#liEmail").value.trim();
      const code = qs("#liCode").value.trim();
      if (!email || !code) return showErr(liErr, "Укажи email и код.");
      try{
        await configReady;
        const vr = await apiFetch(`${API_BASES.v6}/auth/verify-email`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, code })
        });
        const data = await vr.json().catch(()=>({}));
        if (!vr.ok) return showErr(liErr, data.detail || "Код неверный/просрочен");
        liVerifyWrap.style.display = "none";
        toast("Email подтвержден. Теперь выполни вход");
      }catch(e){
        showErr(liErr, "Сервис недоступен");
      }
    });

    qs("#liResendBtn").addEventListener("click", async () => {
      clearErr(liErr);
      const email = qs("#liEmail").value.trim();
      if (!email) return showErr(liErr, "Укажи email.");
      try{
        await configReady;
        const rr = await apiFetch(`${API_BASES.v6}/auth/resend-code`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email })
        });
        const data = await rr.json().catch(()=>({}));
        if (!rr.ok) return showErr(liErr, data.detail || "Не удалось отправить код");
        toast("Код отправлен повторно");
      }catch(e){
        showErr(liErr, "Сервис недоступен");
      }
    });

    qs("#liResetBtn").addEventListener("click", async () => {
      clearErr(liErr);
      const email = qs("#liEmail").value.trim();
      const code = qs("#liResetCode").value.trim();
      const newPass = qs("#liResetPass").value;
      if (!email || !code || !newPass) return showErr(liErr, "Заполни email, код и новый пароль.");
      if (newPass.length < 10) return showErr(liErr, "Новый пароль должен быть минимум 10 символов.");
      try{
        await configReady;
        const r = await apiFetch(`${API_BASES.v6}/auth/reset-password`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, code, new_password: newPass })
        });
        const data = await r.json().catch(()=>({}));
        if (!r.ok) return showErr(liErr, data.detail || "Не удалось сбросить пароль");
        liResetWrap.style.display = "none";
        qs("#liResetCode").value = "";
        qs("#liResetPass").value = "";
        toast("Пароль обновлен. Теперь выполни вход");
      }catch(e){
        showErr(liErr, "Сервис недоступен");
      }
    });

    async function startProdamusPayment(){
      const auth = getAuth();
      if (!auth?.token) {
        toast("Сначала войди в аккаунт");
        openModal("login");
        return;
      }
      try{
        await configReady;
        const r = await apiFetch(`${API_BASES.v6}/billing/prodamus/create-payment`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            plan: selectedPlan.id,
            billing_period: selectedBillingPeriod,
          }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          throw new Error(data?.detail || "Не удалось создать платеж");
        }
        if (!data?.confirmation_url) {
          throw new Error("Prodamus не вернул ссылку на оплату");
        }
        window.location.href = data.confirmation_url;
      }catch(e){
        toast(String(e?.message || e || "Ошибка оплаты"));
      }
    }

    qs("#contactSupportBtn")?.addEventListener("click", () => startProdamusPayment());
    qs("#contactSupportBtn2")?.addEventListener("click", () => startProdamusPayment());

    qs("#skipPayBtn").addEventListener("click", () => {
      toast("Можно оформить позже");
      close();
    });
    qs("#logoutBtn").addEventListener("click", () => {
      clearAuth();
      toast("Вы вышли");
      close();
    });
    qs("#accountLogoutBtn")?.addEventListener("click", () => {
      clearAuth();
      setSub("none");
      toast("Вы вышли");
      close();
    });
    qs("#accountSaveBtn")?.addEventListener("click", () => saveAccountProfile());
    accountCopySupportBtn?.addEventListener("click", () => copySupportPayload());
    qs("#accountPayBtn")?.addEventListener("click", () => startProdamusPayment());
    qs("#accountDemoBtn")?.addEventListener("click", async () => {
      close();
      await requireDemoAccess();
    });
    qs("#logoutTop").addEventListener("click", () => {
      clearAuth();
      setSub("none");
      toast("Вы вышли");
    });

    // Status pill
    function syncUI(){
      const auth = getAuth();
      const sub = getSub();

      if (!auth){
        pillText.textContent = "Public";
        statusPill.title = "Публичный лендинг";
        qs("#loginBtnTop").textContent = "Войти";
        qs("#ctaBtnTop").textContent = "Start trial • 3 days";
        qs("#logoutTop").style.display = "none";
        return;
      }

      if (sub === "active"){
        pillText.textContent = `Pro • ${auth.email}`;
        statusPill.title = "Подписка активна";
        qs("#loginBtnTop").textContent = "Аккаунт";
        qs("#ctaBtnTop").textContent = "Открыть демо";
        qs("#ctaBtnTop").onclick = () => requireDemoAccess();
        qs("#logoutTop").style.display = "inline-flex";
      } else {
        pillText.textContent = `Account • ${auth.email}`;
        statusPill.title = "Нужна подписка";
        qs("#loginBtnTop").textContent = "Аккаунт";
        qs("#ctaBtnTop").textContent = "Choose plan • from 3333 ₽";
        qs("#logoutTop").style.display = "inline-flex";
      }
    }

    // Init
    qs("#year").textContent = String(new Date().getFullYear());
    setSelectedPlan("pro");
    syncUI();
    const _authBoot = getAuth();
    if (_authBoot?.token){
      refreshSubscription();
    }
    completeOauthBridge();
    if (SHOULD_FORCE_LOGIN) {
      setTimeout(() => {
        openModal("login");
        try {
          const url = new URL(window.location.href);
          url.searchParams.delete("login");
          url.searchParams.delete("auth");
          history.replaceState({}, "", url.pathname + (url.search ? url.search : "") + url.hash);
        } catch (_) {}
      }, 60);
    }
    (async () => {
      const uptimeEl = qs("#trustApiUptime");
      const lpApi = qs("#lpApi");
      const lpLatency = qs("#lpLatency");
      const hsMode = qs("#hsMode");
      const hsLatency = qs("#hsLatency");
      const lpChecked = qs("#lpChecked");
      const lpFeed = qs("#lpFeed");
      const lpMode = qs("#lpMode");

      const setChecked = () => {
        const d = new Date();
        const t = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
        if (lpChecked) lpChecked.textContent = t;
      };

      try{
        await configReady;
        const t0 = performance.now();
        const r = await apiFetch(`${API_BASES.v6}/health`, { cache: "no-store" });
        const dt = Math.round(performance.now() - t0);

        // Update values
        if (uptimeEl){
          uptimeEl.textContent = r.ok ? `online • ${dt} ms` : "degraded";
        }
        if (lpLatency) lpLatency.textContent = `${dt} ms`;
          if (hsLatency) hsLatency.textContent = `${dt} ms`;
        if (hsLatency) hsLatency.textContent = `${dt} ms`;
        setChecked();

        if (r.ok){
          if (lpApi) lpApi.textContent = "online";
          if (lpMode) lpMode.textContent = "LIVE";
          if (hsMode) hsMode.textContent = "LIVE";
        }else{
          if (lpApi) lpApi.textContent = "degraded";
          if (lpMode) lpMode.textContent = "DEGRADED";
          if (hsMode) hsMode.textContent = "DEGRADED";
        }

        // Feed remains gated unless user has demo access
        try{
          const auth = getAuth();
          if (auth && auth.token){
            // keep it generic: demo requires subscription check in existing flow
            if (lpFeed) lpFeed.textContent = "available*";
          }
        }catch(_){}

      }catch(e){
        if (uptimeEl) uptimeEl.textContent = "offline";
        if (lpApi) lpApi.textContent = "offline";
        if (lpLatency) lpLatency.textContent = "—";
        if (lpMode) lpMode.textContent = "OFFLINE";
        if (hsMode) hsMode.textContent = "OFFLINE";
        setChecked();
      }
    })();

    // Periodic live-proof refresh
    setInterval(() => {
      try{ (async () => {
        const uptimeEl = qs('#trustApiUptime');
        const lpApi = qs('#lpApi');
        const lpLatency = qs('#lpLatency');
        const lpChecked = qs('#lpChecked');
        const lpMode = qs('#lpMode');
        const hsMode = qs('#hsMode');
        const hsLatency = qs('#hsLatency');
        const t0 = performance.now();
        apiFetch(`${API_BASES.v6}/health`, { cache: 'no-store' }).then(r=>{
          const dt = Math.round(performance.now() - t0);
          const d = new Date();
          const t = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          if (lpChecked) lpChecked.textContent = t;
          if (lpLatency) lpLatency.textContent = `${dt} ms`;
          if (uptimeEl) uptimeEl.textContent = r.ok ? `online • ${dt} ms` : 'degraded';
          if (lpApi) lpApi.textContent = r.ok ? 'online' : 'degraded';
          if (lpMode) lpMode.textContent = r.ok ? 'LIVE' : 'DEGRADED';
          if (hsMode) hsMode.textContent = r.ok ? 'LIVE' : 'DEGRADED';
        }).catch(()=>{
          if (uptimeEl) uptimeEl.textContent = 'offline';
          if (lpApi) lpApi.textContent = 'offline';
          if (lpMode) lpMode.textContent = 'OFFLINE';
          if (hsMode) hsMode.textContent = 'OFFLINE';
        });
      })(); } catch(_){ }
    }, 20000);
;
    // Ensure pricing card highlight is in sync
    try{ setSelectedPlan(selectedPlan?.id || 'pro'); }catch(_){ }
      // Keep pricing highlight aligned on resize
    window.addEventListener('resize', ()=>{
      const btn = document.querySelector(`[data-plan-btn="${selectedPlan?.id || 'pro'}"]`);
      const card = btn ? btn.closest('.price-card') : document.querySelector('.price-card.popular');
      if (card) movePricingHighlight(card);
    }, { passive:true });

  
