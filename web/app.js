/*
 * Déraison Assurances — client web du comparateur d'intent engines.
 *
 * Vanilla JS, modules ES, aucun framework (règle front-ui n°1). Parle à
 * l'API FastAPI (intent_engine/api.py) via fetch. Trois rôles :
 *   1. charger l'état (/api/health) et la base de connaissance (/api/kb) ;
 *   2. envoyer la demande de l'utilisateur (/api/compare ou /api/classify)
 *      et rendre une carte par moteur, plus l'action d'exécution ;
 *   3. la voix : dictée (Web Speech reconnaissance) et lecture (synthèse).
 *
 * Le standard de code s'applique au JS : chaque fonction a un doc-comment,
 * chaque bloc logique est commenté.
 */

// Raccourci DOM. Aucune dépendance externe : $ = document.getElementById.
const $ = (id) => document.getElementById(id);

// Couleur d'accent par moteur, pour teinter les cartes et les barres. Les
// clés correspondent aux noms renvoyés par l'API ; les valeurs sont des
// classes Tailwind (jamais de hex brut dans le markup, règle front-ui n°7).
// Le texte des pastilles (``chip``) utilise des variantes FONCÉES qui
// passent le contraste WCAG AA sur fond teinté clair ; les barres (``bar``)
// utilisent les teintes vives (fond, pas de texte dessus).
const ENGINE_THEME = {
  tfidf: {
    label: '1 · TF-IDF + RandomForest',
    bar: 'bg-sysblue',
    chip: 'text-chipblue bg-sysblue/10',
  },
  fasttext_custom: {
    label: '2 · fastText (appris)',
    bar: 'bg-systeal',
    chip: 'text-chipteal bg-systeal/10',
  },
  fasttext_pretrained: {
    label: '3 · fastText (pré-entraîné)',
    bar: 'bg-sysindigo',
    chip: 'text-chipindigo bg-sysindigo/10',
  },
  bert: {
    label: '4 · BERT + MLP',
    bar: 'bg-sysgreen',
    chip: 'text-chipgreen bg-sysgreen/10',
  },
  llm: {
    label: '5 · LLM',
    bar: 'bg-sysorange',
    chip: 'text-chiporange bg-sysorange/10',
  },
};

// Display order for the comparator columns: the pedagogical progression from
// sparse bag-of-words to generative LLM.
const ENGINE_ORDER = [
  'tfidf',
  'fasttext_custom',
  'fasttext_pretrained',
  'bert',
  'llm',
];

// État applicatif minimal : quels moteurs sont utilisables (renvoyé par
// /api/health) et si une requête est en cours (anti double-soumission).
const state = {
  engines: [],
  busy: false,
};

/**
 * Échappe le HTML pour empêcher toute injection quand on insère du texte
 * utilisateur ou modèle dans le DOM.
 * @param {string} s Texte brut potentiellement dangereux.
 * @returns {string} Texte sûr à interpoler dans du HTML.
 */
const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/**
 * Récupère l'état du serveur et met à jour le badge LLM + la dispo du
 * bouton LLM. Appelé une fois au chargement.
 * @returns {Promise<void>}
 */
async function loadHealth() {
  // Le badge reflète honnêtement si le moteur LLM peut tourner (Ollama up).
  const badge = $('llm-status');
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    state.engines = data.engines || [];
    // Grise tout moteur que l'API ne liste pas comme utilisable maintenant
    // (LLM si Ollama est down, fastText pré-entraîné si le modèle cc.fr.300
    // n'est pas téléchargé) — on ne propose jamais une colonne vouée à échouer.
    disableUnavailableEngines();
    // Le badge d'en-tête reflète l'état du moteur LLM (le plus « fragile »).
    const llmUp = state.engines.includes('llm');
    if (llmUp) {
      // Vert : on affiche le tag du modèle local qui sert le moteur LLM.
      badge.textContent = `LLM en ligne · ${data.llm_model}`;
      badge.className =
        'rounded px-2.5 py-1 text-[12px] font-medium bg-sysgreen/15 text-chipgreen';
    } else {
      badge.textContent = 'LLM hors ligne (Ollama absent)';
      badge.className =
        'rounded px-2.5 py-1 text-[12px] font-medium bg-surface-tertiary ' +
        'text-label-tertiary dark:bg-surface-tertiary-dark';
    }
  } catch (err) {
    // API injoignable : on le dit clairement plutôt que de laisser "…".
    badge.textContent = 'Serveur injoignable';
    badge.className =
      'rounded px-2.5 py-1 text-[12px] font-medium bg-sysorange/15 text-sysorange';
  }
}

/**
 * Grise, dans le sélecteur, tout moteur absent de state.engines (moteurs
 * que l'API ne peut pas exécuter maintenant). Si l'un d'eux était coché, on
 * retombe sur "Comparer tout".
 * @returns {void}
 */
function disableUnavailableEngines() {
  // Parcourt chaque radio de moteur (hors "all", toujours disponible).
  document
    .querySelectorAll('input[name="engine"]')
    .forEach((input) => {
      if (input.value === 'all') return;
      const available = state.engines.includes(input.value);
      // Coupe l'interaction et atténue visuellement l'option indisponible.
      input.disabled = !available;
      const label = input.closest('label');
      if (label) label.classList.toggle('opacity-40', !available);
      if (label) label.classList.toggle('pointer-events-none', !available);
      // Si l'option cochée devient indisponible, on revient à "Comparer tout".
      if (!available && input.checked) {
        document.querySelector('input[name="engine"][value="all"]').checked = true;
      }
    });
}

/**
 * Charge la base de connaissance et rend la liste des intentions, avec des
 * exemples cliquables qui pré-remplissent le champ de saisie.
 * @returns {Promise<void>}
 */
async function loadKnowledgeBase() {
  const res = await fetch('/api/kb');
  const data = await res.json();
  // Compteur affiché dans le <summary> du navigateur de connaissance.
  $('kb-count').textContent = data.count;

  const container = $('kb-list');
  // Une tuile par intention : titre, service, et 3 exemples cliquables.
  container.innerHTML = data.intents
    .map((intent) => {
      // On limite à 3 exemples pour garder les tuiles compactes.
      const chips = intent.examples
        .slice(0, 3)
        .map(
          (ex) =>
            `<button type="button" data-example="${escapeHtml(ex)}"
               class="rounded bg-surface-tertiary px-2 py-1 text-left text-[13px]
                      text-label-secondary hover:bg-sysblue/10 hover:text-sysblue
                      focus:outline-none focus-visible:ring-2 focus-visible:ring-sysblue
                      dark:bg-surface-tertiary-dark dark:text-label-secondary-dark">
               ${escapeHtml(ex)}</button>`,
        )
        .join('');
      return `
        <div class="rounded border border-surface-tertiary p-3
                    dark:border-surface-tertiary-dark">
          <div class="mb-1 flex items-baseline justify-between gap-2">
            <span class="font-medium">${escapeHtml(intent.title)}</span>
            <code class="font-mono text-[12px] text-label-tertiary
                         dark:text-label-tertiary-dark">${escapeHtml(intent.id)}</code>
          </div>
          <p class="mb-2 text-[12px] text-label-tertiary dark:text-label-tertiary-dark">
            → ${escapeHtml(intent.service)}</p>
          <div class="flex flex-wrap gap-1.5">${chips}</div>
        </div>`;
    })
    .join('');

  // Délégation d'événements : un clic sur un exemple le copie dans le champ.
  container.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-example]');
    if (!btn) return;
    $('query').value = btn.dataset.example;
    // On ramène le champ à l'écran et on place le focus pour enchaîner vite.
    $('query').focus();
    $('query').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
}

/**
 * Construit le HTML d'une carte de résultat pour un moteur donné.
 * @param {string} engine Nom du moteur ("tfidf" | "bert" | "llm").
 * @param {object} result Résultat sérialisé renvoyé par l'API.
 * @returns {string} Fragment HTML de la carte.
 */
function renderEngineCard(engine, result) {
  const theme = ENGINE_THEME[engine] || ENGINE_THEME.tfidf;
  const top = result.ranked[0];
  // Latence formatée : on distingue millisecondes et secondes pour le LLM.
  const latency =
    result.latency_ms >= 1000
      ? `${(result.latency_ms / 1000).toFixed(1)} s`
      : `${Math.round(result.latency_ms)} ms`;

  // Cas abstention : aucun top ou confiance insuffisante → transfert humain.
  if (!top || !result.confident) {
    return `
      <article class="rounded bg-surface-secondary p-4 shadow-sm
                      dark:bg-surface-secondary-dark">
        <div class="mb-3 flex items-center justify-between">
          <span class="rounded px-2 py-0.5 text-[12px] font-semibold ${theme.chip}">
            ${theme.label}</span>
          <span class="font-mono text-[12px] text-label-tertiary
                       dark:text-label-tertiary-dark">${latency}</span>
        </div>
        <p class="text-[14px] text-label-secondary dark:text-label-secondary-dark">
          Abstention — intention incertaine, transfert à un conseiller.
        </p>
      </article>`;
  }

  // Barres de confiance pour le top-3 (ou moins). Largeur = score en %.
  const bars = result.ranked
    .slice(0, 3)
    .map((p) => {
      const pct = Math.round(p.score * 100);
      return `
        <div class="mb-1.5">
          <div class="mb-0.5 flex justify-between text-[12px]">
            <span class="font-mono">${escapeHtml(p.intent)}</span>
            <span class="text-label-tertiary dark:text-label-tertiary-dark">${pct}%</span>
          </div>
          <div class="h-1.5 w-full rounded bg-surface-tertiary
                      dark:bg-surface-tertiary-dark">
            <div class="h-1.5 rounded ${theme.bar}" style="width:${pct}%"></div>
          </div>
        </div>`;
    })
    .join('');

  // Slots extraits (moteur LLM) : petits badges clé=valeur si présents.
  const slotsEntries = Object.entries(result.slots || {});
  const slots = slotsEntries.length
    ? `<div class="mt-3 flex flex-wrap gap-1.5">
         ${slotsEntries
           .map(
             ([k, v]) =>
               `<span class="rounded bg-surface-tertiary px-2 py-0.5 text-[12px]
                  dark:bg-surface-tertiary-dark">
                  <span class="text-label-tertiary dark:text-label-tertiary-dark">${escapeHtml(k)}:</span>
                  ${escapeHtml(v)}</span>`,
           )
           .join('')}
       </div>`
    : '';

  // Réponse scriptée rendue en Markdown (marked), texte sérieux en serif.
  const answer = result.response
    ? `<div class="mt-3 border-t border-surface-tertiary pt-3 font-serif text-[14px]
                   leading-relaxed dark:border-surface-tertiary-dark">
         ${window.marked.parse(result.response)}</div>`
    : '';

  return `
    <article class="rounded bg-surface-secondary p-4 shadow-sm
                    dark:bg-surface-secondary-dark">
      <div class="mb-3 flex items-center justify-between">
        <span class="rounded px-2 py-0.5 text-[12px] font-semibold ${theme.chip}">
          ${theme.label}</span>
        <span class="font-mono text-[12px] text-label-tertiary
                     dark:text-label-tertiary-dark">${latency}</span>
      </div>
      ${bars}
      ${slots}
      ${answer}
    </article>`;
}

/**
 * Lit un texte à voix haute via l'API de synthèse vocale du navigateur.
 * C'est le "speech-helper" : le bot répond aussi à l'oral.
 * @param {string} text Texte à prononcer (français).
 * @returns {void}
 */
function speak(text) {
  // Garde-fou : API absente (vieux navigateur) → on ne fait rien.
  if (!('speechSynthesis' in window) || !text) return;
  // On annule toute lecture en cours pour ne pas empiler les voix.
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  // Voix française : la réponse client est en français.
  utter.lang = 'fr-FR';
  window.speechSynthesis.speak(utter);
}

/**
 * Envoie la demande à l'API, rend les cartes moteur et l'action exécutée.
 * @param {Event} event Événement submit du formulaire.
 * @returns {Promise<void>}
 */
async function onSubmit(event) {
  event.preventDefault();
  // Anti double-clic pendant qu'une requête (potentiellement lente, LLM) tourne.
  if (state.busy) return;
  const text = $('query').value.trim();
  if (!text) return;

  // Quel moteur ? "all" = comparateur, sinon un seul.
  const engine = document.querySelector('input[name="engine"]:checked').value;
  const results = $('results');
  const execution = $('execution');

  // État occupé : on désactive le bouton et on affiche des squelettes.
  state.busy = true;
  $('submit-btn').disabled = true;
  $('submit-btn').textContent = 'Analyse en cours…';
  execution.classList.add('hidden');
  // Squelettes de chargement (règle front-ui : feedback > 300 ms).
  results.innerHTML = skeletonCards(engine);

  try {
    // Comparateur : /api/compare renvoie tous les moteurs d'un coup.
    if (engine === 'all') {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      // Ordre d'affichage stable : la progression pédagogique (croissant en
      // coût/sophistication), en ne gardant que les moteurs renvoyés.
      const order = ENGINE_ORDER.filter((e) => data[e]);
      results.innerHTML = order
        .map((e) => renderEngineCard(e, data[e]))
        .join('');
      // Lecture vocale : on lit la réponse du meilleur moteur disponible.
      maybeSpeakBest(order.map((e) => data[e]));
    } else {
      // Moteur unique : /api/classify.
      const res = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, engine }),
      });
      const data = await res.json();
      results.innerHTML = renderEngineCard(engine, data);
      maybeSpeakBest([data]);
    }

    // Action concrète : on "exécute" la demande avec le moteur choisi. En
    // mode comparateur, on prend BERT s'il est disponible (le moteur rapide
    // le plus exact) plutôt que le défaut serveur, pour une action pertinente.
    const execEngine =
      engine === 'all'
        ? state.engines.includes('bert')
          ? 'bert'
          : null
        : engine;
    await renderExecution(text, execEngine);
  } catch (err) {
    // Erreur réseau : message clair dans la zone de résultats.
    results.innerHTML =
      '<p class="text-sysorange">Erreur : impossible de contacter le serveur.</p>';
  } finally {
    // Retour à l'état interactif quoi qu'il arrive.
    state.busy = false;
    $('submit-btn').disabled = false;
    $('submit-btn').textContent = "Analyser l'intention";
  }
}

/**
 * Rend des cartes "squelette" pendant le chargement.
 * @param {string} engine Moteur choisi ("all" ou un nom précis).
 * @returns {string} HTML des squelettes.
 */
function skeletonCards(engine) {
  // Nombre de squelettes = nombre de colonnes attendues.
  const count = engine === 'all' ? state.engines.length || 3 : 1;
  const one = `
    <article class="animate-pulse rounded bg-surface-secondary p-4 shadow-sm
                    dark:bg-surface-secondary-dark motion-reduce:animate-none">
      <div class="mb-3 h-4 w-24 rounded bg-surface-tertiary
                  dark:bg-surface-tertiary-dark"></div>
      <div class="mb-2 h-2 w-full rounded bg-surface-tertiary
                  dark:bg-surface-tertiary-dark"></div>
      <div class="h-2 w-2/3 rounded bg-surface-tertiary
                  dark:bg-surface-tertiary-dark"></div>
    </article>`;
  return one.repeat(count);
}

/**
 * Appelle /api/execute et affiche l'action d'aiguillage (service, action,
 * slots) ou le transfert vers un humain.
 * @param {string} text Demande de l'utilisateur.
 * @param {string|null} engine Moteur à utiliser, ou null pour le défaut.
 * @returns {Promise<void>}
 */
async function renderExecution(text, engine) {
  const box = $('execution');
  const res = await fetch('/api/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, engine }),
  });
  const exec = await res.json();
  box.classList.remove('hidden');

  // Cas transfert humain : le filet « Je ne sais pas » → conseiller non-IA.
  // On affiche le message d'aveu d'incertitude renvoyé par le serveur, plus
  // le service d'escalade, pour que le repli soit explicite et rassurant.
  if (exec.handoff) {
    const msg =
      exec.message ||
      "Je préfère vous passer un conseiller plutôt que de vous orienter au hasard.";
    box.innerHTML = `
      <p class="mb-1 text-[15px] font-semibold">→ Je ne sais pas — transfert à un conseiller humain</p>
      <p class="mb-2 font-serif text-[14px] leading-relaxed
                text-label-secondary dark:text-label-secondary-dark">
        ${escapeHtml(msg)}</p>
      <p class="text-[13px] text-label-tertiary dark:text-label-tertiary-dark">
        Service : ${escapeHtml(exec.service || "Accueil téléphonique")}</p>`;
    return;
  }

  // Cas action : on montre ce qu'un système aval (CRM, SVI) recevrait.
  const slots = Object.entries(exec.slots || {})
    .map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(v)}`)
    .join(', ');
  box.innerHTML = `
    <p class="mb-1 text-[15px] font-semibold">→ Action exécutée</p>
    <dl class="grid gap-1 text-[14px] sm:grid-cols-2">
      <div><dt class="inline text-label-tertiary dark:text-label-tertiary-dark">Intention :</dt>
        <dd class="inline font-medium">${escapeHtml(exec.title)}</dd></div>
      <div><dt class="inline text-label-tertiary dark:text-label-tertiary-dark">Service :</dt>
        <dd class="inline">${escapeHtml(exec.service)}</dd></div>
      <div><dt class="inline text-label-tertiary dark:text-label-tertiary-dark">Action :</dt>
        <dd class="inline font-mono">${escapeHtml(exec.action)}</dd></div>
      ${slots ? `<div><dt class="inline text-label-tertiary dark:text-label-tertiary-dark">Infos :</dt>
        <dd class="inline font-mono">${slots}</dd></div>` : ''}
    </dl>`;
}

/**
 * Lit à voix haute la meilleure réponse disponible si l'option est cochée.
 * @param {object[]} resultList Résultats moteurs, ordre de préférence.
 * @returns {void}
 */
function maybeSpeakBest(resultList) {
  // On ne parle que si l'utilisateur a activé la lecture vocale.
  if (!$('speak-toggle').checked) return;
  // On prend la première réponse non vide (moteur le plus fiable en tête).
  const withAnswer = resultList.find((r) => r && r.response);
  if (withAnswer) speak(withAnswer.response);
}

/**
 * Initialise la dictée vocale (Web Speech reconnaissance) sur le bouton
 * micro. C'est le "vocal-helper" : l'utilisateur parle, le texte s'écrit.
 * @returns {void}
 */
function setupMic() {
  const btn = $('mic-btn');
  // Compat : l'API est préfixée webkit sur Safari/Chrome.
  const Recognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  // Navigateur sans reconnaissance vocale : on masque le bouton proprement.
  if (!Recognition) {
    btn.classList.add('hidden');
    return;
  }
  const recognizer = new Recognition();
  // Français, pas de résultats intermédiaires : on veut la phrase finale.
  recognizer.lang = 'fr-FR';
  recognizer.interimResults = false;

  // Clic : démarre l'écoute et signale l'état par la couleur du bouton.
  btn.addEventListener('click', () => {
    recognizer.start();
    btn.classList.add('bg-sysblue', 'text-white');
  });
  // Résultat : on écrit la transcription dans le champ de saisie.
  recognizer.addEventListener('result', (e) => {
    $('query').value = e.results[0][0].transcript;
  });
  // Fin/erreur : on rétablit l'apparence normale du bouton.
  const reset = () => btn.classList.remove('bg-sysblue', 'text-white');
  recognizer.addEventListener('end', reset);
  recognizer.addEventListener('error', reset);
}

/**
 * Point d'entrée : câble les événements et charge l'état initial.
 * @returns {void}
 */
function init() {
  // Formulaire principal.
  $('ask-form').addEventListener('submit', onSubmit);
  // Voix (dictée).
  setupMic();
  // Chargements réseau en parallèle : état serveur + base de connaissance.
  loadHealth();
  loadKnowledgeBase();
}

// Démarrage une fois le DOM prêt (le script est un module, donc différé).
init();
