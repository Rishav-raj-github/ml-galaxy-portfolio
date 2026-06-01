/**
 * ML Galaxy Portfolio - Front-End Controller Engine (2026 Edition)
 * Controls routing, search indexes, canvas sandboxes, Simulated IDE toggles,
 * and the interactive Mock Interview quiz simulator.
 */

// Application Global States
let activeModel = null;
let activeTab = 1; // 1 = From Scratch, 2 = Applied
let activeSandbox = null;

// Mock Interview Simulator States
let quizQuestions = [];
let currentQuizIdx = 0;
let prepStatusDb = {}; // Model ID -> 'poor', 'good', 'perfect'

document.addEventListener("DOMContentLoaded", () => {
    initializeExplorer();
    initializePrepChecklist();
    loadQuizQuestions();
    loadNextQuizQuestion();
});

// --- Tab Routing Management ---
function showSection(sectionId) {
    document.querySelectorAll(".section-pane").forEach(pane => pane.classList.remove("active"));
    document.querySelectorAll(".nav-btn").forEach(btn => btn.classList.remove("active"));
    
    document.getElementById(sectionId).classList.add("active");
    
    // Set matching navigation active
    const btnMap = {
        'explorer-sec': 0,
        'interview-sec': 1
    };
    document.querySelectorAll(".nav-btn")[btnMap[sectionId]].classList.add("active");
}

// --- Explorer & Directory Render Engine ---
function initializeExplorer() {
    const container = document.getElementById("categories-container");
    if (!container) return;
    container.innerHTML = "";

    Object.keys(MODEL_DATA).forEach(catKey => {
        const cat = MODEL_DATA[catKey];
        
        const catCard = document.createElement("div");
        catCard.className = "category-card glass-card animate-slide-up";
        catCard.style.borderLeftColor = catKey === 'regression' ? 'var(--accent-cyan)' :
                                      catKey === 'classification' ? 'var(--accent-red)' :
                                      catKey === 'unsupervised' ? 'var(--accent-purple)' :
                                      catKey === 'deeplearning' ? 'var(--accent-pink)' : 'var(--accent-green)';

        catCard.innerHTML = `
            <div class="category-title-bar">
                <span class="category-icon">${cat.icon}</span>
                <h3>${cat.title}</h3>
            </div>
            <p class="category-desc">${cat.description}</p>
            <div class="tiles-list" id="tiles-list-${catKey}"></div>
        `;
        container.appendChild(catCard);

        const listContainer = document.getElementById(`tiles-list-${catKey}`);
        cat.models.forEach(model => {
            const tile = document.createElement("div");
            tile.className = "model-tile";
            tile.id = `tile-${model.id}`;
            tile.onclick = () => selectModel(model.id);

            const star = model.important2026 ? `<span class="star-badge" title="2026 Job Critical">⭐</span>` : '';
            const badgeText = model.badge ? `<span class="cat-tag">${model.badge}</span>` : '';

            tile.innerHTML = `
                <span class="tile-name">${model.name}</span>
                <div class="tile-badges">
                    ${badgeText}
                    ${star}
                </div>
            `;
            listContainer.appendChild(tile);
        });
    });

    updateResultsCount();
}

// --- Dynamic Search Index Filters ---
function filterModels() {
    const query = document.getElementById("search-input").value.toLowerCase();
    const priorityOnly = document.getElementById("priority-toggle").checked;

    let totalVisible = 0;

    Object.keys(MODEL_DATA).forEach(catKey => {
        const cat = MODEL_DATA[catKey];
        let catVisibleCount = 0;

        cat.models.forEach(model => {
            const tile = document.getElementById(`tile-${model.id}`);
            if (!tile) return;

            const matchesQuery = model.name.toLowerCase().includes(query) || 
                                 model.concept.toLowerCase().includes(query);
            const matchesPriority = !priorityOnly || model.important2026;

            if (matchesQuery && matchesPriority) {
                tile.style.display = "flex";
                catVisibleCount++;
                totalVisible++;
            } else {
                tile.style.display = "none";
            }
        });

        // Hide whole category card if empty
        const catCard = document.getElementById(`tiles-list-${catKey}`).parentNode;
        catCard.style.display = catVisibleCount > 0 ? "block" : "none";
    });

    updateResultsCount(totalVisible, query || priorityOnly);
}

function updateResultsCount(count, isFiltered) {
    const badge = document.getElementById("results-count");
    if (!badge) return;

    if (count === undefined) {
        let total = 0;
        Object.keys(MODEL_DATA).forEach(k => total += MODEL_DATA[k].models.length);
        badge.innerText = `Total Models: ${total}`;
    } else {
        badge.innerText = isFiltered ? `Found: ${count}` : `Total Models: ${count}`;
    }
}

// --- Detail Drawer Routing ---
function selectModel(modelId) {
    // 1. Locate matching model object
    let modelObj = null;
    let categoryTitle = "";
    Object.keys(MODEL_DATA).forEach(k => {
        const found = MODEL_DATA[k].models.find(m => m.id === modelId);
        if (found) {
            modelObj = found;
            categoryTitle = MODEL_DATA[k].title;
        }
    });

    if (!modelObj) return;
    activeModel = modelObj;

    // Remove old active classes
    document.querySelectorAll(".model-tile").forEach(t => t.classList.remove("active"));
    const tile = document.getElementById(`tile-${modelId}`);
    if (tile) tile.classList.add("active");

    // Reveal Panels
    document.getElementById("placeholder-panel").style.display = "none";
    const detailPanel = document.getElementById("detail-panel");
    detailPanel.style.display = "block";
    detailPanel.className = "detail-card glass-card animate-fade-in"; // trigger re-anim

    // Populate Fields
    document.getElementById("detail-title").innerText = modelObj.name;
    document.getElementById("detail-category-tag").innerText = categoryTitle;
    document.getElementById("detail-concept").innerText = modelObj.concept;
    document.getElementById("detail-industry-badge").style.display = modelObj.important2026 ? "block" : "none";

    // Mathematical Render using KaTeX auto-renderer
    const mathEl = document.getElementById("detail-math");
    mathEl.innerHTML = modelObj.math;
    if (window.renderMathInElement) {
        renderMathInElement(mathEl, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    }

    // Load Pros vs Cons
    const prosContainer = document.getElementById("detail-pros");
    const consContainer = document.getElementById("detail-cons");
    prosContainer.innerHTML = modelObj.pros.map(p => `<li>${p}</li>`).join("");
    consContainer.innerHTML = modelObj.cons.map(c => `<li>${c}</li>`).join("");

    // Load IDE Code base
    activeTab = 1; // Default back to theory
    updateCodeEditor();

    // Load Flashcard Accordions
    const cardsContainer = document.getElementById("detail-flashcards");
    cardsContainer.innerHTML = "";
    modelObj.qna.forEach((qna, idx) => {
        const card = document.createElement("div");
        card.className = "flashcard";
        card.innerHTML = `
            <div class="flashcard-q" onclick="toggleFlashcard(this)">
                <span>Q: ${qna.q}</span>
                <span class="flashcard-icon">▶</span>
            </div>
            <div class="flashcard-a">
                <p>${qna.a}</p>
            </div>
        `;
        cardsContainer.appendChild(card);
    });

    // --- Launch Visual Canvas Sandboxes ---
    setupSandbox(modelObj.id);
}

function toggleFlashcard(el) {
    const card = el.parentNode;
    card.classList.toggle("open");
}

// --- Companion IDE Controller ---
function updateCodeEditor() {
    if (!activeModel) return;
    
    const tab1 = document.getElementById("tab-proj1");
    const tab2 = document.getElementById("tab-proj2");
    
    if (activeTab === 1) {
        tab1.classList.add("active");
        tab2.classList.remove("active");
        
        document.getElementById("ide-filepath").innerText = activeModel.project1.file;
        document.getElementById("ide-code-content").innerText = activeModel.project1.code;
    } else {
        tab1.classList.remove("active");
        tab2.classList.add("active");
        
        document.getElementById("ide-filepath").innerText = activeModel.project2.file;
        document.getElementById("ide-code-content").innerText = activeModel.project2.code;
    }
}

function switchProjectTab(tabIdx) {
    activeTab = tabIdx;
    updateCodeEditor();
}

function copySourceCode() {
    const code = document.getElementById("ide-code-content").innerText;
    navigator.clipboard.writeText(code).then(() => {
        const copyBtn = document.querySelector(".copy-btn");
        copyBtn.innerText = "✓ Copied!";
        setTimeout(() => copyBtn.innerText = "Copy Code", 1500);
    });
}

// --- HTML5 Sandbox Visual Router ---
function setupSandbox(modelId) {
    const container = document.getElementById("sandbox-container");
    const canvas = document.getElementById("sandbox-canvas");
    
    // Hide controls initially
    document.getElementById("regression-controls").style.display = "none";
    document.getElementById("kmeans-controls").style.display = "none";
    document.getElementById("neural-controls").style.display = "none";

    // Terminate existing canvas animation bindings if any
    activeSandbox = null;

    if (modelId === "linear_regression") {
        container.style.display = "block";
        document.getElementById("regression-controls").style.display = "flex";
        
        // Reset defaults
        document.getElementById("reg-degree").value = 1;
        document.getElementById("degree-val").innerText = 1;
        document.getElementById("reg-lambda").value = 0;
        document.getElementById("lambda-val").innerText = "0.0";

        activeSandbox = new RegressionSandbox("sandbox-canvas");
    } else if (modelId === "decision_trees") {
        container.style.display = "block";
        document.getElementById("kmeans-controls").style.display = "flex";
        
        // Let's use KMeans clustering engine for decision trees cluster sandbox
        document.getElementById("kmeans-k").value = 3;
        document.getElementById("k-val").innerText = 3;
        
        activeSandbox = new KMeansSandbox("sandbox-canvas");
    } else if (modelId === "mlp_attention") {
        container.style.display = "block";
        document.getElementById("neural-controls").style.display = "flex";
        
        activeSandbox = new NeuralNetworkSandbox("sandbox-canvas");
    } else {
        // No visualizer mapped
        container.style.display = "none";
    }
}

// --- Sandbox Action Forwarders ---
function updateRegressionParams() {
    if (!activeSandbox || !(activeSandbox instanceof RegressionSandbox)) return;
    const d = parseInt(document.getElementById("reg-degree").value);
    const l = parseFloat(document.getElementById("reg-lambda").value) / 100.0;

    document.getElementById("degree-val").innerText = d;
    document.getElementById("lambda-val").innerText = l.toFixed(2);

    activeSandbox.setDegree(d);
    activeSandbox.setLambda(l);
}

function clearRegressionPoints() {
    if (activeSandbox && activeSandbox instanceof RegressionSandbox) {
        activeSandbox.clear();
    }
}

function updateKMeansParams() {
    if (!activeSandbox || !(activeSandbox instanceof KMeansSandbox)) return;
    const k = document.getElementById("kmeans-k").value;
    document.getElementById("k-val").innerText = k;
    activeSandbox.setK(k);
}

function stepKMeans() {
    if (activeSandbox && activeSandbox instanceof KMeansSandbox) {
        activeSandbox.step();
    }
}

function regenerateKMeansPoints() {
    if (activeSandbox && activeSandbox instanceof KMeansSandbox) {
        activeSandbox.generateRandomPoints(60);
    }
}

function propagateNeuralNetwork() {
    if (activeSandbox && activeSandbox instanceof NeuralNetworkSandbox) {
        activeSandbox.propagate();
    }
}


// ==========================================
// 🤝 SECTION 2: INTERVIEW PREPARATION MODULE
// ==========================================

function initializePrepChecklist() {
    const checklist = document.getElementById("prep-checklist-container");
    if (!checklist) return;
    checklist.innerHTML = "";

    // Load Core 20 models
    let checklistIdx = 1;
    Object.keys(MODEL_DATA).forEach(catKey => {
        MODEL_DATA[catKey].models.forEach(m => {
            // Register all models in local progress database
            if (!prepStatusDb[m.id]) {
                prepStatusDb[m.id] = 'poor'; // default
            }

            const item = document.createElement("div");
            item.className = "prep-item";
            item.innerHTML = `
                <span class="prep-name">${checklistIdx++}. ${m.name}</span>
                <select class="prep-status-select poor" id="prep-select-${m.id}" onchange="changePrepStatus('${m.id}', this.value)">
                    <option value="poor">Needs Review ❌</option>
                    <option value="good">Good Progress 🟨</option>
                    <option value="perfect">Mastered 🏆</option>
                </select>
            `;
            checklist.appendChild(item);
        });
    });
}

function changePrepStatus(modelId, val) {
    prepStatusDb[modelId] = val;
    const select = document.getElementById(`prep-select-${modelId}`);
    if (select) {
        select.className = `prep-status-select ${val}`;
    }
}

function loadQuizQuestions() {
    quizQuestions = [];
    Object.keys(MODEL_DATA).forEach(catKey => {
        MODEL_DATA[catKey].models.forEach(m => {
            m.qna.forEach(qna => {
                quizQuestions.push({
                    modelId: m.id,
                    modelName: m.name,
                    category: MODEL_DATA[catKey].title,
                    question: qna.q,
                    expertAnswer: qna.a
                });
            });
        });
    });

    // Shuffle questions
    quizQuestions.sort(() => Math.random() - 0.5);
}

function loadNextQuizQuestion() {
    if (quizQuestions.length === 0) return;
    
    // Increment index
    currentQuizIdx = (currentQuizIdx + 1) % quizQuestions.length;
    const q = quizQuestions[currentQuizIdx];

    document.getElementById("quiz-model-category").innerText = `${q.category} / ${q.modelName}`;
    document.getElementById("quiz-question-text").innerText = q.question;
    document.getElementById("quiz-expert-answer").innerText = q.expertAnswer;

    // Collapse answer panel
    document.getElementById("quiz-answer-panel").style.display = "none";
}

function toggleQuizAnswer() {
    const panel = document.getElementById("quiz-answer-panel");
    panel.style.display = panel.style.display === "none" ? "block" : "none";
}

function scoreSelfProgress(scoreType) {
    const q = quizQuestions[currentQuizIdx];
    
    // Update checklist status dropdown value mapping
    const statusMap = {
        'poor': 'poor',
        'good': 'good',
        'perfect': 'perfect'
    };
    
    changePrepStatus(q.modelId, statusMap[scoreType]);
    
    const checklistSelect = document.getElementById(`prep-select-${q.modelId}`);
    if (checklistSelect) {
        checklistSelect.value = statusMap[scoreType];
        checklistSelect.className = `prep-status-select ${statusMap[scoreType]}`;
    }

    // Auto load next question with a slight success flash animation
    const card = document.getElementById("quiz-card-box");
    card.style.borderColor = scoreType === 'perfect' ? 'rgba(15, 240, 179, 0.4)' :
                             scoreType === 'good' ? 'rgba(255, 177, 153, 0.4)' : 'rgba(255, 8, 68, 0.4)';
    
    setTimeout(() => {
        card.style.borderColor = 'rgba(255,255,255,0.04)';
        loadNextQuizQuestion();
    }, 800);
}
