const API_URL = "/api/search";
const IMAGE_SEARCH_URL = "/api/search-image";
const PARSE_URL = "/api/parse-query";
const COCO_URL = "/api/coco-classes";
const TIMELINE_URL = "/api/video-timeline";
const RERANK_URL = "/api/rerank-top-videos";
const MULTI_SCENE_URL = "/api/search-multi-scene";
const QWEN_RERANK_CHAINS_URL = "/api/qwen-rerank-chains";
const QWEN_STATUS_URL = "/api/qwen-status";
const READINESS_URL = "/api/readiness";
const READINESS_RETRY_URL = "/api/readiness/retry";
const VIDEO_FPS_API = "/api/video-fps";
const KEYFRAME_RESOLVE_API = "/api/keyframe-resolve";
const SUBMISSION_FILL_URL = "/api/submission/fill";
const SUBMISSION_SAVE_URL = "/api/submission/save-file";
const SUBMISSION_LIST_URL = "/api/submission/list-files";
const SUBMISSION_ZIP_URL = "/api/submission/export-zip";

// Tính năng phụ vẫn được giữ ở backend nhưng ẩn khỏi luồng thao tác chung kết.
const OCR_FEATURE_VISIBLE = false;
const RERANK_FEATURE_VISIBLE = false;

// Image Query Elements
const cardImageQuery = document.getElementById("card-image-query");
const imageDropzone = document.getElementById("image-dropzone");
const imageDropzonePlaceholder = document.getElementById("image-dropzone-placeholder");
const imageFileInput = document.getElementById("image-file-input");
const imagePreviewContainer = document.getElementById("image-preview-container");
const imagePreviewImg = document.getElementById("image-preview-img");
const btnRemoveImageQuery = document.getElementById("btn-remove-image-query");
const btnSearchImageQuery = document.getElementById("btn-search-image-query");
let currentImageQueryBase64 = null;

// Submission Manager Elements
const cardSubmission = document.getElementById("card-submission");
const subDirPathInput = document.getElementById("sub-dir-path");
const btnSubRefreshDir = document.getElementById("btn-sub-refresh-dir");
const subFileListContainer = document.getElementById("sub-file-list-container");
const subFileCountSpan = document.getElementById("sub-file-count");
const subFileItemsContainer = document.getElementById("sub-file-items");

const subModePills = document.getElementById("sub-mode-pills");
let currentSubMode = "kis";

const subVideoIdInput = document.getElementById("sub-video-id");
const subKisQaFields = document.getElementById("sub-kis-qa-fields");
const subFrameAInput = document.getElementById("sub-frame-a");
const subFrameBInput = document.getElementById("sub-frame-b");
const subFrameA2Input = document.getElementById("sub-frame-a2");
const subFrameB2Input = document.getElementById("sub-frame-b2");

const subQaFields = document.getElementById("sub-qa-fields");
const subQaAnswerInput = document.getElementById("sub-qa-answer");

const subTrakeFields = document.getElementById("sub-trake-fields");
const subTrakeFramesInput = document.getElementById("sub-trake-frames");

const subEnableHedgingCheck = document.getElementById("sub-enable-hedging");
const subFileNameInput = document.getElementById("sub-file-name");
const btnSubFillPreview = document.getElementById("btn-sub-fill-preview");
const btnSubSaveFile = document.getElementById("btn-sub-save-file");

const subPreviewContainer = document.getElementById("sub-preview-container");
const subPreviewTotalSpan = document.getElementById("sub-preview-total");
const subPreviewStatusSpan = document.getElementById("sub-preview-status");
const subPreviewCode = document.getElementById("sub-preview-code");

const subZipOutputPathInput = document.getElementById("sub-zip-output-path");
const btnSubExportZip = document.getElementById("btn-sub-export-zip");
const lightboxSubBtn = document.getElementById("lightbox-sub-btn");

let latestGeneratedSubmissionLines = [];

// Mode Switcher Elements
const modeKisBtn = document.getElementById("mode-kis-btn");
const modeTrakeBtn = document.getElementById("mode-trake-btn");
const kisActionDock = document.getElementById("kis-action-dock");
const trakeActionDock = document.getElementById("trake-action-dock");

// ASR Context & Guidance Elements
const asrWindowPills = document.getElementById("asr-window-pills");
let selectedAsrWindowN = 1;

const geminiRerankGuidance = document.getElementById("gemini-rerank-guidance");
const headerRerankGuidance = document.getElementById("header-rerank-guidance");
const bodyRerankGuidance = document.getElementById("body-rerank-guidance");
const chevronRerankGuidance = document.getElementById("chevron-rerank-guidance");
const discriminatingTagsContainer = document.getElementById("discriminating-tags-container");
const addDiscFeatureInput = document.getElementById("add-disc-feature-input");
const btnAddDiscFeature = document.getElementById("btn-add-disc-feature");
const rerankHintInput = document.getElementById("rerank-hint-input");
let currentDiscriminatingFeatures = [];

// TRAKE Qwen Rerank Toolbar Elements
const trakeQwenToolbar = document.getElementById("trake-qwen-toolbar");
const trakeQwenRerankBtn = document.getElementById("trake-qwen-rerank-btn");
const qwenTopkPills = document.getElementById("qwen-topk-pills");
const qwenTopkChainsInput = document.getElementById("qwen-topk-chains-input");
const qwenProgressText = document.getElementById("qwen-progress-text");
const qwenStatusBadge = document.getElementById("qwen-status-badge");

// TRAKE Parameter Elements
const cardTrakeParams = document.getElementById("card-trake-params");
const trakeCandidatePoolSlider = document.getElementById("trake-candidate-pool-slider");
const trakeCandidatePoolInput = document.getElementById("trake-candidate-pool");
const trakeTopVInput = document.getElementById("trake-top-v-input");
const trakeFramesPerVidInput = document.getElementById("trake-frames-per-vid");
const trakeLambdaInput = document.getElementById("trake-lambda-input");
const trakeKpathsInput = document.getElementById("trake-kpaths-input");
const trakeTopKInput = document.getElementById("trake-top-k-input");
const trakeSingleSearchBtn = document.getElementById("trake-single-search-btn");
const trakeSearchBtn = document.getElementById("trake-search-btn");
const trakeExportBtn = document.getElementById("trake-export-btn");
const resultsMainTitle = document.getElementById("results-main-title");
const multiSceneBadgeTitle = document.getElementById("multi-scene-badge-title");

// Elements - Parser & Multi-Scene
const rawQueryInput = document.getElementById("raw-query-input");
const aiParseBtn = document.getElementById("ai-parse-btn");
const geminiApiKeyInput = document.getElementById("gemini-api-key");
const qaCallout = document.getElementById("qa-callout");
const qaQuestionText = document.getElementById("qa-question-text");
const multiSceneBar = document.getElementById("multi-scene-bar");
const multiSceneSummary = document.getElementById("multi-scene-summary");
const multiSceneTabs = document.getElementById("multi-scene-tabs");
const multiSceneEditorCards = document.getElementById("multi-scene-editor-cards");
const toggleSceneEditorBtn = document.getElementById("toggle-scene-editor-btn");

// Fusion Strategy (RRF)
const useRrfCheck = document.getElementById("use-rrf-check");
const rrfKInput = document.getElementById("rrf-k-input");
const rrfKGroup = document.getElementById("rrf-k-group");
const fusionModeDesc = document.getElementById("fusion-mode-desc");

// Module 1: Visual
const enableVisualCheck = document.getElementById("enable-visual-check");
const bodyVisual = document.getElementById("body-visual");
const searchInput = document.getElementById("search-input");
const weightVisualSlider = document.getElementById("weight-visual");
const valWeightVisual = document.getElementById("val-weight-visual");

// Module 2: ASR
const enableAsrCheck = document.getElementById("enable-asr-check");
const bodyAsr = document.getElementById("body-asr");
const speechInput = document.getElementById("speech-input");
const weightAsrSlider = document.getElementById("weight-asr");
const valWeightAsr = document.getElementById("val-weight-asr");

// Module 3: OCR
const enableOcrCheck = document.getElementById("enable-ocr-check");
const bodyOcr = document.getElementById("body-ocr");
const ocrInput = document.getElementById("ocr-input");
const weightOcrSlider = document.getElementById("weight-ocr");
const valWeightOcr = document.getElementById("val-weight-ocr");

// Module 4: RT-DETR
const enableObjectCheck = document.getElementById("enable-object-check");
const bodyObject = document.getElementById("body-object");
const objectClassSelect = document.getElementById("object-class-select");
const objectConfSlider = document.getElementById("object-conf-slider");
const valObjectConf = document.getElementById("val-object-conf");
const weightObjectSlider = document.getElementById("weight-object");
const valWeightObject = document.getElementById("val-weight-object");

// Module 5: Smart Dedup
const smartDedupCheck = document.getElementById("smart-dedup-check");
const bodyDedup = document.getElementById("body-dedup");
const visualSimSlider = document.getElementById("visual-sim-slider");
const visualSimNum = document.getElementById("visual-sim-num");
const uniqueVideoCheck = document.getElementById("unique-video-check");

// Module 6: Candidates & Rerank Filter
const topKSlider = document.getElementById("top-k-slider");
const topKNum = document.getElementById("top-k-num");
const rerankTopV = document.getElementById("rerank-top-v");
const rerankKPerVideo = document.getElementById("rerank-k-per-video");
const labelK2Total = document.getElementById("label-k2-total");
const btnTopVLabel = document.getElementById("btn-top-v-label");
const btnK2Label = document.getElementById("btn-k2-label");
const videoIdInput = document.getElementById("video-id");
const filterVideoResultsInput = document.getElementById("filter-video-results");

// Module 6A: Danh Mục & Tags Filter Elements (Database Pushdown)
const CATEGORIES_URL = "/api/categories";
const cardCategoryFilter = document.getElementById("card-category-filter");
const categorySelect = document.getElementById("category-select");
const groupSubcategory = document.getElementById("group-subcategory");
const subcategorySelect = document.getElementById("subcategory-select");
const clearCategoryFilterBtn = document.getElementById("clear-category-filter");
const includeTagsContainer = document.getElementById("include-tags-container");
const includeVideoInput = document.getElementById("include-video-input");
const excludeTagsContainer = document.getElementById("exclude-tags-container");
const excludeVideoInput = document.getElementById("exclude-video-input");

let currentIncludeTags = [];
let currentExcludeTags = [];
let categoriesTaxonomy = {};
let pendingCategorySelection = [];
let pendingSubcategorySelection = [];

// Đặt bộ lọc ngay dưới Visual Query mà không nhân đôi markup trong sidebar nhỏ.
const visualCardForLayout = document.getElementById("card-visual");
if (visualCardForLayout && cardCategoryFilter) {
    visualCardForLayout.insertAdjacentElement("afterend", cardCategoryFilter);
}

// Cross-Video Dedup Elements (Chống QC/Intro lặp lại)
const enableCrossVideoDedupCheck = document.getElementById("enable-cross-video-dedup-check");
const crossVideoSimSlider = document.getElementById("cross-video-sim-slider");
const crossVideoSimNum = document.getElementById("cross-video-sim-num");

// Search & Rerank Action Buttons
const searchBtn = document.getElementById("search-btn");
const multiSearchBtn = document.getElementById("multi-search-btn");
const rerankBtn = document.getElementById("rerank-btn");
const statusText = document.getElementById("status-text");
const resultsGrid = document.getElementById("results-grid");
const resultCount = document.getElementById("result-count");
const systemReadiness = document.getElementById("system-readiness");
const systemReadinessText = document.getElementById("system-readiness-text");

let backendReady = false;
let readinessPollTimer = null;

function setSearchControlsReady(ready) {
    backendReady = ready;
    const hasRrfSignal = !useRrfCheck?.checked
        || Boolean(rrfSignalVisual?.checked || rrfSignalAsr?.checked || rrfSignalShot?.checked || rrfSignalCaption?.checked);
    if (searchBtn) searchBtn.disabled = !ready || !hasRrfSignal;
    if (trakeSingleSearchBtn) trakeSingleSearchBtn.disabled = !ready;
    if (trakeSearchBtn) trakeSearchBtn.disabled = !ready;
    if (multiSearchBtn) multiSearchBtn.disabled = !ready || !isMultiSceneMode;
}

function renderReadinessState(data) {
    const state = data?.state || "starting";
    const progress = Number.isFinite(Number(data?.progress)) ? Number(data.progress) : 0;
    const isReady = state === "ready";
    setSearchControlsReady(isReady);

    if (!systemReadiness || !systemReadinessText) return;
    systemReadiness.classList.remove("readiness-warming", "readiness-ready", "readiness-error");

    if (isReady) {
        systemReadiness.classList.add("readiness-ready");
        systemReadinessText.innerText = "READY";
        systemReadiness.title = `Sẵn sàng sau ${data.elapsed_sec || 0}s`;
        if (readinessPollTimer) {
            clearInterval(readinessPollTimer);
            readinessPollTimer = null;
        }
    } else if (state === "error") {
        systemReadiness.classList.add("readiness-error");
        systemReadinessText.innerText = "WARM-UP ERROR";
        systemReadiness.title = `${data.error || data.message || "Warm-up lỗi"} — click để thử lại`;
    } else {
        systemReadiness.classList.add("readiness-warming");
        systemReadinessText.innerText = `WARMING UP ${progress}%`;
        systemReadiness.title = data.message || "Backend đang warm-up";
    }
}

async function fetchSystemReadiness() {
    try {
        const response = await fetch(READINESS_URL, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        renderReadinessState(payload.data || {});
    } catch (err) {
        renderReadinessState({ state: "error", error: `Không đọc được readiness: ${err.message}` });
    }
}

async function retrySystemWarmup() {
    if (!systemReadiness?.classList.contains("readiness-error")) return;
    renderReadinessState({ state: "warming", progress: 0, message: "Đang thử warm-up lại..." });
    try {
        await fetch(READINESS_RETRY_URL, { method: "POST" });
        if (!readinessPollTimer) readinessPollTimer = setInterval(fetchSystemReadiness, 750);
        await fetchSystemReadiness();
    } catch (err) {
        renderReadinessState({ state: "error", error: err.message });
    }
}

if (systemReadiness) systemReadiness.addEventListener("click", retrySystemWarmup);

// RRF Signal Selector Elements
const rrfSignalVisual = document.getElementById("rrf-signal-visual");
const rrfSignalAsr = document.getElementById("rrf-signal-asr");
const rrfSignalShot = document.getElementById("rrf-signal-shot");
const rrfSignalCaption = document.getElementById("rrf-signal-caption");

if (rrfSignalVisual && localStorage.getItem("aic_rrf_visual") !== null) {
    rrfSignalVisual.checked = localStorage.getItem("aic_rrf_visual") === "true";
}
if (rrfSignalAsr && localStorage.getItem("aic_rrf_asr") !== null) {
    rrfSignalAsr.checked = localStorage.getItem("aic_rrf_asr") === "true";
}
if (rrfSignalShot && localStorage.getItem("aic_rrf_shot") !== null) {
    rrfSignalShot.checked = localStorage.getItem("aic_rrf_shot") === "true";
}
if (rrfSignalCaption && localStorage.getItem("aic_rrf_caption") !== null) {
    rrfSignalCaption.checked = localStorage.getItem("aic_rrf_caption") === "true";
}

[rrfSignalVisual, rrfSignalAsr, rrfSignalShot, rrfSignalCaption].forEach(el => {
    if (el) {
        el.addEventListener("change", () => {
            if (rrfSignalVisual) localStorage.setItem("aic_rrf_visual", rrfSignalVisual.checked);
            if (rrfSignalAsr) localStorage.setItem("aic_rrf_asr", rrfSignalAsr.checked);
            if (rrfSignalShot) localStorage.setItem("aic_rrf_shot", rrfSignalShot.checked);
            if (rrfSignalCaption) localStorage.setItem("aic_rrf_caption", rrfSignalCaption.checked);
            validateRrfSignals();
        });
    }
});

function validateRrfSignals() {
    const v = rrfSignalVisual ? rrfSignalVisual.checked : true;
    const a = rrfSignalAsr ? rrfSignalAsr.checked : true;
    const s = rrfSignalShot ? rrfSignalShot.checked : true;
    const c = rrfSignalCaption ? rrfSignalCaption.checked : false;
    if (!v && !a && !s && !c) {
        showToast("⚠️ Vui lòng bật ít nhất 1 tín hiệu RRF!", "warning");
        if (searchBtn) searchBtn.disabled = true;
    } else {
        if (searchBtn) searchBtn.disabled = !backendReady;
    }
}

// ==========================================================================
// SMART STICKY HEADER (Tự động ẩn khi lướt xuống, hiện ngay khi kéo nhẹ lên)
// ==========================================================================
const mainContentEl = document.querySelector(".main-content");
const resultsHeaderPro = document.querySelector(".results-header-pro");

if (mainContentEl && resultsHeaderPro) {
    let lastScrollTop = 0;
    const SCROLL_DELTA_THRESHOLD = 8; // Ngưỡng 8px chống rung giật khi cuộn chậm

    mainContentEl.addEventListener("scroll", () => {
        const st = mainContentEl.scrollTop;

        // Nếu ở gần đỉnh (<= 20px), luôn luôn giữ thanh hiển thị bình thường
        if (st <= 20) {
            resultsHeaderPro.classList.remove("header-hidden");
            lastScrollTop = st;
            return;
        }

        const diff = st - lastScrollTop;
        if (Math.abs(diff) < SCROLL_DELTA_THRESHOLD) {
            return;
        }

        if (diff > 0) {
            // Lướt xuống (Scroll Down) -> Ẩn thanh header để giải phóng 100% không gian soi video
            resultsHeaderPro.classList.add("header-hidden");
        } else {
            // Kéo nhẹ lên (Scroll Up) -> Trượt xuống ngay tức thì để người dùng đổi chế độ/xem Top-K
            resultsHeaderPro.classList.remove("header-hidden");
        }

        lastScrollTop = st;
    }, { passive: true });
}

// Lightbox Elements
const lightboxModal = document.getElementById("lightbox-modal");
const lightboxCloseBtn = document.getElementById("lightbox-close-btn");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCanvas = document.getElementById("lightbox-canvas");
const lightboxToggleBoxes = document.getElementById("lightbox-toggle-boxes");
const lightboxTitle = document.getElementById("lightbox-title");
const lightboxScoreBadge = document.getElementById("lightbox-score-badge");
const lightboxTimeBadge = document.getElementById("lightbox-time-badge");
const lightboxCopyBtn = document.getElementById("lightbox-copy-btn");
const lightboxTimelineBtn = document.getElementById("lightbox-timeline-btn");
const lightboxVideoBtn = document.getElementById("lightbox-video-btn");
const copyStatus = document.getElementById("copy-status");
const lightboxTranscript = document.getElementById("lightbox-transcript");
const lightboxBoxesList = document.getElementById("lightbox-boxes-list");

const lightboxCanvasControls = document.getElementById("lightbox-canvas-controls");

// Timeline Elements
const timelineModal = document.getElementById("timeline-modal");
const timelineVideoTitle = document.getElementById("timeline-video-title");
const timelineFrameCount = document.getElementById("timeline-frame-count");
const timelineCloseBtn = document.getElementById("timeline-close-btn");
const timelineStrip = document.getElementById("timeline-strip");

// Video Player Elements
const VIDEO_STREAM_URL = "/videos";
const videoModal = document.getElementById("video-modal");
const videoPlayerTitle = document.getElementById("video-player-title");
const videoTimeBadge = document.getElementById("video-time-badge");
const videoCloseBtn = document.getElementById("video-close-btn");
const html5VideoPlayer = document.getElementById("html5-video-player");
const videoRewind5s = document.getElementById("video-rewind-5s");
const videoJumpKeyframe = document.getElementById("video-jump-keyframe");
const videoForward5s = document.getElementById("video-forward-5s");
const videoSpeedSelect = document.getElementById("video-speed-select");
const videoCopyBtn = document.getElementById("video-copy-btn");
const videoTrakeSubmitBtn = document.getElementById("video-trake-submit-btn");
const videoModeHint = document.getElementById("video-mode-hint");
const quickVideoOverlay = document.getElementById("quick-video-overlay");
const quickVideoOpenBtn = document.getElementById("quick-video-open-btn");
const quickVideoCloseBtn = document.getElementById("quick-video-close");
const quickVideoIdInput = document.getElementById("quick-video-id-input");
const quickVideoTimeInput = document.getElementById("quick-video-time-input");
const btnQuickVideoGo = document.getElementById("btn-quick-video-go");
const btnQuickVideoTimeline = document.getElementById("btn-quick-video-timeline");
const btnQuickVideoSubmit = document.getElementById("btn-quick-video-submit");
const quickSubmitMode = document.getElementById("quick-submit-mode");
const quickVideoMapStatus = document.getElementById("quick-video-map-status");
const quickVideoTabs = document.querySelectorAll(".quick-video-tab");
const quickVideoVerifyPanel = document.getElementById("quick-video-verify-panel");
const quickVideoSubmitPanel = document.getElementById("quick-video-submit-panel");

// Global state
let currentResults = [];
let currentLightboxIndex = 0;
let activeResultsSource = [];
let cocoClasses = [];
let latestRerankerQuery = "";
let latestUnifiedVisual = "";
let latestUnifiedSpeech = "";
let latestUnifiedOcr = "";
let latestParsedScenes = [];
let isMultiSceneMode = false;
let multiSceneResultsData = null;
let activeSceneTabIdx = -1;
let currentPlayingTimestamp = 0;
let currentPlayingVideoId = "";
let currentPlayingFrameId = "";
const _videoFpsCache = Object.create(null);
const _videoFpsRequests = Object.create(null);
let _videoBadgeRafId = null;
let _lastBadgeUpdate = 0;
let _videoOutlineTimer = null;

async function getVideoFpsWithCache(videoId) {
    const normalizedId = String(videoId || "").trim().toUpperCase();
    if (!normalizedId) return 25.0;
    if (Object.prototype.hasOwnProperty.call(_videoFpsCache, normalizedId)) {
        return _videoFpsCache[normalizedId];
    }
    if (_videoFpsRequests[normalizedId]) return _videoFpsRequests[normalizedId];

    _videoFpsRequests[normalizedId] = (async () => {
        try {
            const response = await fetch(`${VIDEO_FPS_API}/${encodeURIComponent(normalizedId)}`);
            if (!response.ok) throw new Error(`FPS API HTTP ${response.status}`);
            const payload = await response.json();
            const fps = Number(payload.fps);
            _videoFpsCache[normalizedId] = Number.isFinite(fps) && fps > 0 ? fps : 25.0;
        } catch (error) {
            console.warn(`[VideoPlayer] Dùng FPS mặc định cho ${normalizedId}:`, error);
            _videoFpsCache[normalizedId] = 25.0;
        } finally {
            delete _videoFpsRequests[normalizedId];
        }
        return _videoFpsCache[normalizedId];
    })();

    return _videoFpsRequests[normalizedId];
}

// TRAKE Mode States
let currentAppMode = "KIS"; // "KIS" | "TRAKE"
let latestTrakeSequences = [];
let latestTrakeSubmissionLines = [];
let trakeEnterStage = 0;
let trakeDraftSignature = "";
let trakeDraftSourceId = "";

// ==========================================================================
// CHUYỂN ĐỔI CHẾ ĐỘ TÌM KIẾM: KIS/MULTIMODAL <--> TRAKE
// State này độc lập hoàn toàn với dresTaskModeSelect (thể thức nộp bài).
// ==========================================================================
function switchMode(mode, saveState = true) {
    currentAppMode = mode;

    if (mode === "TRAKE") {
        if (modeTrakeBtn) modeTrakeBtn.classList.add("active");
        if (modeKisBtn) modeKisBtn.classList.remove("active");

        if (kisActionDock) kisActionDock.style.display = "none";
        if (trakeActionDock) trakeActionDock.style.display = "flex";

        if (cardTrakeParams) {
            cardTrakeParams.style.display = "block";
            cardTrakeParams.classList.remove("is-collapsed");
            const body = document.getElementById("body-trake-params");
            if (body) body.style.display = "block";
            const chevron = cardTrakeParams.querySelector(".chevron-icon");
            if (chevron) chevron.innerText = "▲";
        }

        if (resultsMainTitle) {
            resultsMainTitle.innerText = "⏱️ TRAKE STORYBOARD MATRIX (TARS MONOTONIC DP)";
        }

        // Khôi phục draft khi người dùng quay lại tab TRAKE. Trước đây editor
        // chỉ được render đúng lúc parse nên bị trống sau một vòng KIS -> TRAKE.
        if (latestParsedScenes && latestParsedScenes.length >= 2) {
            const geminiCard = document.getElementById("card-gemini");
            setCardCollapsed(geminiCard, false);
            if (multiSceneBar) multiSceneBar.style.display = "flex";
            if (multiSceneBadgeTitle) multiSceneBadgeTitle.innerText = `⏱️ TRAKE (${latestParsedScenes.length} EVENTS)`;
            renderMultiSceneTabs(latestParsedScenes);
            const draftIndex = activeSceneTabIdx >= 0 ? activeSceneTabIdx : 0;
            renderTrakeEventEditor(draftIndex);
        }

        // Nếu đã có kết quả TRAKE trước đó -> render lại storyboard keyframe.
        if (latestTrakeSequences && latestTrakeSequences.length > 0) {
            renderTrakeStoryboard(latestTrakeSequences);
        }

        showToast("⏱️ Đã chuyển sang chế độ TRAKE (Temporal Event Extraction)!", "info", 2000);
    } else {
        // Rời TRAKE nhưng giữ latestParsedScenes làm draft. Bỏ con trỏ editor
        // để thao tác Visual Query ở KIS không âm thầm ghi đè một event TRAKE.
        activeSceneTabIdx = -1;
        if (modeKisBtn) modeKisBtn.classList.add("active");
        if (modeTrakeBtn) modeTrakeBtn.classList.remove("active");

        if (kisActionDock) kisActionDock.style.display = "flex";
        if (trakeActionDock) trakeActionDock.style.display = "none";

        if (cardTrakeParams) {
            cardTrakeParams.style.display = "none";
        }
        if (multiSceneEditorCards) multiSceneEditorCards.style.display = "none";

        if (resultsMainTitle) {
            resultsMainTitle.innerText = "MULTIMODAL RESULTS";
        }

        // Render lại kết quả KIS bằng keyframe tĩnh.
        if (currentResults && currentResults.length > 0) {
            renderResults(currentResults);
        }

        showToast("🔍 Đã chuyển về chế độ KIS / QA thông thường.", "info", 1800);
    }

    if (saveState) {
        saveControlPanelState();
    }
}

if (modeKisBtn) {
    modeKisBtn.addEventListener("click", () => switchMode("KIS"));
}
if (modeTrakeBtn) {
    modeTrakeBtn.addEventListener("click", () => switchMode("TRAKE"));
}

// ==========================================================================
// 0. FLOATING TOAST NOTIFICATION SYSTEM
// ==========================================================================
let activeLoadingToast = null;

function showToast(message, type = "info", duration = 3500) {
    const container = document.getElementById("toast-container");
    if (!container) return null;

    if (type === "loading" && activeLoadingToast) {
        activeLoadingToast.remove();
        activeLoadingToast = null;
    }

    const toast = document.createElement("div");
    toast.className = `toast-item toast-${type}`;

    let icon = "ℹ️";
    if (type === "success") icon = "✅";
    else if (type === "error") icon = "❌";
    else if (type === "warning") icon = "⚠️";
    else if (type === "loading") icon = "⏳";

    const iconElement = document.createElement("span");
    iconElement.className = "toast-icon";
    iconElement.textContent = icon;
    const messageElement = document.createElement("span");
    messageElement.className = "toast-message";
    messageElement.textContent = String(message ?? "");
    toast.append(iconElement, messageElement);
    container.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.add("show");
    });

    if (type === "loading") {
        activeLoadingToast = toast;
        return toast;
    }

    if (duration > 0) {
        setTimeout(() => {
            toast.classList.remove("show");
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 300);
        }, duration);
    }
    return toast;
}

function dismissLoadingToast() {
    if (activeLoadingToast) {
        activeLoadingToast.classList.remove("show");
        setTimeout(() => {
            if (activeLoadingToast && activeLoadingToast.parentNode) {
                activeLoadingToast.parentNode.removeChild(activeLoadingToast);
            }
            activeLoadingToast = null;
        }, 250);
    }
}

// ==========================================================================
// 1. COLLAPSE / EXPAND TOGGLE & RRF STATE HANDLERS
// ==========================================================================
function setCardCollapsed(cardEl, collapse) {
    if (!cardEl) return;
    if (collapse) {
        cardEl.classList.add("is-collapsed");
    } else {
        cardEl.classList.remove("is-collapsed");
    }
    const chevron = cardEl.querySelector(".chevron-icon");
    if (chevron) {
        chevron.innerText = collapse ? "▼" : "▲";
    }
}

function toggleCardCollapse(cardEl) {
    if (!cardEl) return;
    const isNowCollapsed = !cardEl.classList.contains("is-collapsed");
    setCardCollapsed(cardEl, isNowCollapsed);
    saveControlPanelState();
}

// Bắt sự kiện click vào header card
document.querySelectorAll(".feature-card").forEach(card => {
    const header = card.querySelector(".card-header");
    if (header) {
        header.addEventListener("click", (e) => {
            if (e.target.closest(".switch-small") || e.target.closest("input[type='checkbox']")) {
                return;
            }
            toggleCardCollapse(card);
        });
    }
});

// Đóng / Xổ Khung Đặc Trưng Phân Biệt (Qwen Guide)
function setGuidanceCollapsed(collapse) {
    if (!geminiRerankGuidance || !bodyRerankGuidance) return;
    if (collapse) {
        geminiRerankGuidance.classList.add("is-collapsed");
        bodyRerankGuidance.style.display = "none";
        if (chevronRerankGuidance) chevronRerankGuidance.innerText = "▼";
    } else {
        geminiRerankGuidance.classList.remove("is-collapsed");
        bodyRerankGuidance.style.display = "block";
        if (chevronRerankGuidance) chevronRerankGuidance.innerText = "▲";
    }
}

function toggleGuidanceCollapse() {
    if (!geminiRerankGuidance) return;
    const isNowCollapsed = !geminiRerankGuidance.classList.contains("is-collapsed");
    setGuidanceCollapsed(isNowCollapsed);
    saveControlPanelState();
}

if (headerRerankGuidance) {
    headerRerankGuidance.addEventListener("click", toggleGuidanceCollapse);
}

// Logic RRF Mode UI Toggle
function updateRrfUIState() {
    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const kVal = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;

    if (rrfKGroup) {
        if (isRrf) {
            rrfKGroup.classList.remove("disabled-k");
            if (rrfKInput) rrfKInput.disabled = false;
        } else {
            rrfKGroup.classList.add("disabled-k");
            if (rrfKInput) rrfKInput.disabled = true;
        }
    }

    // Vô hiệu hóa triệt để tất cả slider trọng số khi bật RRF
    const sliders = [weightVisualSlider, weightOcrSlider, weightAsrSlider, weightObjectSlider];
    sliders.forEach(slider => {
        if (slider) slider.disabled = isRrf;
    });

    const weightRows = document.querySelectorAll(".weight-row");
    weightRows.forEach(row => {
        if (isRrf) {
            row.classList.add("disabled-weights");
        } else {
            row.classList.remove("disabled-weights");
        }
    });
}

if (useRrfCheck) {
    useRrfCheck.addEventListener("change", () => {
        updateRrfUIState();
        saveControlPanelState();
        showToast(useRrfCheck.checked ? `⚡ Đã bật RRF (k=${rrfKInput.value || 60}) - Trọng số đã khóa` : "⚖️ Đã chuyển sang Weighted Fusion - Cho phép kéo trọng số", "info", 2200);
    });
}

if (rrfKInput) {
    rrfKInput.addEventListener("input", () => {
        saveControlPanelState();
    });
}

// Bắt sự kiện Checkbox tính năng -> Tự động thu / xổ card tương ứng
function bindAutoCollapse(checkEl, cardId) {
    if (!checkEl) return;
    checkEl.addEventListener("change", () => {
        const card = document.getElementById(cardId);
        if (card) {
            setCardCollapsed(card, !checkEl.checked);
            saveControlPanelState();
        }
    });
}

bindAutoCollapse(enableVisualCheck, "card-visual");
bindAutoCollapse(enableAsrCheck, "card-asr");
bindAutoCollapse(enableOcrCheck, "card-ocr");
bindAutoCollapse(enableObjectCheck, "card-object");
bindAutoCollapse(smartDedupCheck, "card-dedup");

// ==========================================================================
// 2. STATE PERSISTENCE (LOCAL STORAGE - BAO GỒM TOÀN BỘ THAM SỐ TARS TRAKE)
// ==========================================================================
const STORAGE_KEY = "aic_2026_control_panel_state_v3";

function getSelectedValues(selectEl) {
    return selectEl ? Array.from(selectEl.selectedOptions).map(option => option.value).filter(Boolean) : [];
}

function applySelectedValues(selectEl, values) {
    if (!selectEl) return;
    const selected = new Set(Array.isArray(values) ? values : []);
    Array.from(selectEl.options).forEach(option => {
        option.selected = selected.has(option.value);
    });
}

function getTaxonomyFilterPayload() {
    const categories = getSelectedValues(categorySelect);
    const subjects = categories.includes("day_hoc") ? getSelectedValues(subcategorySelect) : [];
    return { categories, subjects };
}

const RRF_CHANNEL_UI = {
    visual: { icon: "👁️", short: "V", label: "Visual", rankKey: "rank_vis" },
    asr: { icon: "🎙️", short: "A", label: "ASR", rankKey: "rank_asr" },
    shot: { icon: "🎬", short: "S", label: "Shot", rankKey: "rank_shot" },
    caption: { icon: "📝", short: "C", label: "Caption", rankKey: "rank_caption" },
};

function rrfChannelState(item, channel) {
    const meta = RRF_CHANNEL_UI[channel];
    const activeMap = item.rrf_active_channels || {};
    const fallbackActive = {
        visual: Boolean(useRrfCheck?.checked && rrfSignalVisual?.checked && enableVisualCheck?.checked),
        asr: Boolean(useRrfCheck?.checked && rrfSignalAsr?.checked && enableAsrCheck?.checked),
        shot: Boolean(useRrfCheck?.checked && rrfSignalShot?.checked),
        caption: Boolean(useRrfCheck?.checked && rrfSignalCaption?.checked),
    }[channel];
    const rrfEnabled = item.rrf_enabled ?? Boolean(useRrfCheck?.checked);
    const available = Array.isArray(item.rrf_available_channels)
        ? item.rrf_available_channels.includes(channel)
        : ({
            visual: true,
            asr: Boolean(item.asr_transcript || item[meta.rankKey] != null),
            shot: Boolean(item.shot_id || item[meta.rankKey] != null),
            caption: Boolean(item.caption_available),
        }[channel]);
    const active = rrfEnabled && Boolean(activeMap[channel] ?? fallbackActive);
    const rank = item[meta.rankKey];
    if (!active) return { value: "OFF", state: "off", title: `${meta.label}: không được bật cho truy vấn RRF này` };
    if (!available) return { value: "N/A", state: "na", title: `${meta.label}: keyframe/shot này không có dữ liệu của kênh` };
    if (rank == null) return { value: "MISS", state: "miss", title: `${meta.label}: kênh đã chạy và có dữ liệu, nhưng không lọt danh sách hit` };
    const contribution = Number(item.rrf_contributions?.[channel] || 0);
    const suffix = contribution > 0 ? ` · đóng góp raw ${contribution.toFixed(4)}` : "";
    return { value: `#${rank}`, state: "hit", title: `${meta.label}: đang tham gia RRF ở hạng #${rank}${suffix}` };
}

function renderRrfChannelBadge(item, channel, verbose = false) {
    const meta = RRF_CHANNEL_UI[channel];
    const status = rrfChannelState(item, channel);
    const name = verbose ? `${meta.label}: ` : `${meta.short} `;
    return `<span class="rank-badge badge-${channel} is-${status.state}" title="${status.title}">${meta.icon} ${name}${status.value}</span>`;
}

function saveControlPanelState() {
    try {
        const state = {
            app_mode: currentAppMode,
            use_rrf: useRrfCheck ? useRrfCheck.checked : true,
            rrf_k: rrfKInput ? parseInt(rrfKInput.value) || 60 : 60,
            enable_visual: enableVisualCheck ? enableVisualCheck.checked : true,
            enable_ocr: enableOcrCheck ? enableOcrCheck.checked : false,
            enable_asr: enableAsrCheck ? enableAsrCheck.checked : true,
            enable_object: enableObjectCheck ? enableObjectCheck.checked : false,
            object_class: objectClassSelect ? objectClassSelect.value : "0",
            object_conf: objectConfSlider ? objectConfSlider.value : "0.30",
            weight_visual: weightVisualSlider ? weightVisualSlider.value : "1.0",
            weight_ocr: weightOcrSlider ? weightOcrSlider.value : "1.0",
            weight_asr: weightAsrSlider ? weightAsrSlider.value : "1.0",
            weight_object: weightObjectSlider ? weightObjectSlider.value : "0.5",
            smart_dedup: smartDedupCheck ? smartDedupCheck.checked : true,
            smart_submode: (document.querySelector('input[name="smart-submode"]:checked') || {}).value || "smart",
            visual_sim: visualSimSlider ? visualSimSlider.value : "0.90",
            unique_video: uniqueVideoCheck ? uniqueVideoCheck.checked : false,
            enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
            cross_video_sim: crossVideoSimSlider ? crossVideoSimSlider.value : "0.90",
            asr_window_n: selectedAsrWindowN,
            top_k: topKNum ? topKNum.value : "100",
            top_v: rerankTopV ? rerankTopV.value : "5",
            k_per_video: rerankKPerVideo ? rerankKPerVideo.value : "40",
            categories: getSelectedValues(categorySelect),
            sub_categories: getSelectedValues(subcategorySelect),

            // Tham số TARS DP (TRAKE Mode)
            trake_candidate_pool: trakeCandidatePoolInput ? trakeCandidatePoolInput.value : "100",
            trake_top_v: trakeTopVInput ? trakeTopVInput.value : "10",
            trake_frames_per_vid: trakeFramesPerVidInput ? trakeFramesPerVidInput.value : "40",
            trake_lambda: trakeLambdaInput ? trakeLambdaInput.value : "0.001",
            trake_kpaths: trakeKpathsInput ? trakeKpathsInput.value : "3",
            trake_top_k: trakeTopKInput ? trakeTopKInput.value : "10",

            // Trạng thái thu gọn (Collapsed States)
            collapsed_gemini: document.getElementById("card-gemini")?.classList.contains("is-collapsed"),
            collapsed_image_query: document.getElementById("card-image-query")?.classList.contains("is-collapsed"),
            collapsed_rerank_guidance: geminiRerankGuidance?.classList.contains("is-collapsed"),
            collapsed_visual: document.getElementById("card-visual")?.classList.contains("is-collapsed"),
            collapsed_asr: document.getElementById("card-asr")?.classList.contains("is-collapsed"),
            collapsed_ocr: document.getElementById("card-ocr")?.classList.contains("is-collapsed"),
            collapsed_object: document.getElementById("card-object")?.classList.contains("is-collapsed"),
            collapsed_dedup: document.getElementById("card-dedup")?.classList.contains("is-collapsed"),
            collapsed_category: cardCategoryFilter?.classList.contains("is-collapsed"),
            collapsed_filter: document.getElementById("card-filter")?.classList.contains("is-collapsed"),
            collapsed_trake_params: document.getElementById("card-trake-params")?.classList.contains("is-collapsed"),
            collapsed_submission: document.getElementById("card-submission")?.classList.contains("is-collapsed"),
            sub_dir_path: subDirPathInput ? subDirPathInput.value : "query/query-p1/submission",
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
        console.warn("Could not save to localStorage", e);
    }
}

function restoreBoundedNumberInput(input, savedValue) {
    if (!input) return savedValue;
    const parsed = Number(savedValue);
    const min = Number(input.min);
    const max = Number(input.max);
    const fallback = Number(input.defaultValue || input.value);
    const inRange = Number.isFinite(parsed)
        && (!Number.isFinite(min) || parsed >= min)
        && (!Number.isFinite(max) || parsed <= max);
    const restored = inRange ? parsed : fallback;
    input.value = restored;
    return restored;
}

function getBoundedNumberInput(input, fallback) {
    if (!input) return fallback;
    const parsed = Number(input.value);
    const min = Number(input.min);
    const max = Number(input.max);
    const safeValue = Number.isFinite(parsed) ? parsed : fallback;
    const bounded = Math.min(
        Number.isFinite(max) ? max : safeValue,
        Math.max(Number.isFinite(min) ? min : safeValue, safeValue)
    );
    input.value = bounded;
    return bounded;
}

function restoreControlPanelState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            // Mặc định ban đầu: Gemini, Visual và ASR MỞ; OCR, Object, Dedup, Filter, Submission THU GỌN
            setCardCollapsed(document.getElementById("card-gemini"), false);
            setCardCollapsed(document.getElementById("card-visual"), false);
            setCardCollapsed(document.getElementById("card-asr"), false);
            setCardCollapsed(document.getElementById("card-ocr"), true);
            setCardCollapsed(document.getElementById("card-object"), true);
            setCardCollapsed(document.getElementById("card-dedup"), true);
            setCardCollapsed(document.getElementById("card-filter"), true);
            setCardCollapsed(document.getElementById("card-submission"), true);
            updateRrfUIState();
            return;
        }
        const state = JSON.parse(raw);
        pendingCategorySelection = Array.isArray(state.categories) ? state.categories : [];
        pendingSubcategorySelection = Array.isArray(state.sub_categories) ? state.sub_categories : [];
        if (state.use_rrf !== undefined && useRrfCheck) useRrfCheck.checked = state.use_rrf;
        if (state.rrf_k !== undefined && rrfKInput) rrfKInput.value = state.rrf_k;

        if (state.enable_visual !== undefined && enableVisualCheck) enableVisualCheck.checked = state.enable_visual;
        if (state.enable_ocr !== undefined && enableOcrCheck) enableOcrCheck.checked = state.enable_ocr;
        if (state.enable_asr !== undefined && enableAsrCheck) enableAsrCheck.checked = state.enable_asr;
        if (enableObjectCheck) enableObjectCheck.checked = false;

        if (state.asr_window_n !== undefined && asrWindowPills) {
            selectedAsrWindowN = parseInt(state.asr_window_n) || 1;
            asrWindowPills.querySelectorAll(".btn-pill").forEach(btn => {
                if (parseInt(btn.getAttribute("data-n")) === selectedAsrWindowN) {
                    btn.classList.add("active");
                } else {
                    btn.classList.remove("active");
                }
            });
        }

        if (state.object_class !== undefined && objectClassSelect) objectClassSelect.value = state.object_class;
        if (state.object_conf !== undefined && objectConfSlider) {
            objectConfSlider.value = state.object_conf;
            if (valObjectConf) valObjectConf.innerText = Number(state.object_conf).toFixed(2);
        }

        if (state.weight_visual !== undefined && weightVisualSlider) {
            weightVisualSlider.value = state.weight_visual;
            if (valWeightVisual) valWeightVisual.innerText = Number(state.weight_visual).toFixed(1);
        }
        if (state.weight_ocr !== undefined && weightOcrSlider) {
            weightOcrSlider.value = state.weight_ocr;
            if (valWeightOcr) valWeightOcr.innerText = Number(state.weight_ocr).toFixed(1);
        }
        if (state.weight_asr !== undefined && weightAsrSlider) {
            weightAsrSlider.value = state.weight_asr;
            if (valWeightAsr) valWeightAsr.innerText = Number(state.weight_asr).toFixed(1);
        }
        if (state.weight_object !== undefined && weightObjectSlider) {
            weightObjectSlider.value = state.weight_object;
            if (valWeightObject) valWeightObject.innerText = Number(state.weight_object).toFixed(1);
        }

        if (state.smart_dedup !== undefined && smartDedupCheck) smartDedupCheck.checked = state.smart_dedup;
        if (state.smart_submode) {
            const rad = document.querySelector(`input[name="smart-submode"][value="${state.smart_submode}"]`);
            if (rad) rad.checked = true;
        }
        if (state.visual_sim !== undefined && visualSimSlider) {
            visualSimSlider.value = state.visual_sim;
            if (visualSimNum) visualSimNum.value = state.visual_sim;
        }
        if (state.unique_video !== undefined && uniqueVideoCheck) uniqueVideoCheck.checked = state.unique_video;

        if (state.enable_cross_video_dedup !== undefined && enableCrossVideoDedupCheck) enableCrossVideoDedupCheck.checked = state.enable_cross_video_dedup;
        if (state.cross_video_sim !== undefined && crossVideoSimSlider) {
            crossVideoSimSlider.value = state.cross_video_sim;
            if (crossVideoSimNum) crossVideoSimNum.value = state.cross_video_sim;
        }

        if (state.top_k !== undefined && topKNum) {
            topKNum.value = state.top_k;
            if (topKSlider) topKSlider.value = state.top_k;
        }
        if (state.top_v !== undefined && rerankTopV) rerankTopV.value = state.top_v;
        if (state.k_per_video !== undefined && rerankKPerVideo) rerankKPerVideo.value = state.k_per_video;

        // Khôi phục Tham số TARS DP (TRAKE)
        if (state.trake_candidate_pool !== undefined && trakeCandidatePoolInput) {
            trakeCandidatePoolInput.value = state.trake_candidate_pool;
            if (trakeCandidatePoolSlider) trakeCandidatePoolSlider.value = state.trake_candidate_pool;
        }
        if (state.trake_top_v !== undefined && trakeTopVInput) {
            state.trake_top_v = restoreBoundedNumberInput(trakeTopVInput, state.trake_top_v);
        }
        if (state.trake_frames_per_vid !== undefined && trakeFramesPerVidInput) {
            state.trake_frames_per_vid = restoreBoundedNumberInput(trakeFramesPerVidInput, state.trake_frames_per_vid);
        }
        if (state.trake_lambda !== undefined && trakeLambdaInput) trakeLambdaInput.value = state.trake_lambda;
        if (state.trake_kpaths !== undefined && trakeKpathsInput) trakeKpathsInput.value = state.trake_kpaths;
        if (state.trake_top_k !== undefined && trakeTopKInput) trakeTopKInput.value = state.trake_top_k;

        // Loai bo gia tri cu vuot min/max de lan F5 sau khong tai lai cau hinh nang.
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));

        if (state.sub_dir_path && subDirPathInput) {
            subDirPathInput.value = state.sub_dir_path;
        }

        // Khôi phục Chế độ KIS / TRAKE (không lưu đè state khi đang restore)
        if (state.app_mode && state.app_mode !== currentAppMode) {
            switchMode(state.app_mode, false);
        }

        // Khôi phục collapsed state (Gemini luôn mở trừ khi user cố tình thu)
        setCardCollapsed(document.getElementById("card-gemini"), state.collapsed_gemini !== undefined ? state.collapsed_gemini : false);
        setCardCollapsed(document.getElementById("card-visual"), state.collapsed_visual !== undefined ? state.collapsed_visual : !enableVisualCheck.checked);
        setCardCollapsed(document.getElementById("card-asr"), state.collapsed_asr !== undefined ? state.collapsed_asr : !enableAsrCheck.checked);
        setCardCollapsed(document.getElementById("card-ocr"), state.collapsed_ocr !== undefined ? state.collapsed_ocr : !enableOcrCheck.checked);
        setCardCollapsed(document.getElementById("card-object"), state.collapsed_object !== undefined ? state.collapsed_object : !enableObjectCheck.checked);
        setCardCollapsed(document.getElementById("card-dedup"), state.collapsed_dedup !== undefined ? state.collapsed_dedup : !smartDedupCheck.checked);
        setCardCollapsed(cardCategoryFilter, state.collapsed_category !== undefined ? state.collapsed_category : true);
        setCardCollapsed(document.getElementById("card-filter"), state.collapsed_filter !== undefined ? state.collapsed_filter : true);
        if (cardTrakeParams && state.collapsed_trake_params !== undefined) {
            setCardCollapsed(cardTrakeParams, state.collapsed_trake_params);
        }
        setCardCollapsed(document.getElementById("card-submission"), state.collapsed_submission !== undefined ? state.collapsed_submission : true);

        if (state.collapsed_rerank_guidance !== undefined && geminiRerankGuidance) {
            setRerankGuidanceCollapsed(state.collapsed_rerank_guidance);
        }

        updateRrfUIState();
        updateK2Calculation();
    } catch (e) {
        console.warn("Could not restore localStorage", e);
    }
}

// Sliders and Values Sync
if (objectConfSlider && valObjectConf) {
    objectConfSlider.addEventListener("input", (e) => {
        valObjectConf.innerText = parseFloat(e.target.value).toFixed(2);
        saveControlPanelState();
    });
}

if (weightObjectSlider && valWeightObject) {
    weightObjectSlider.addEventListener("input", (e) => {
        valWeightObject.innerText = parseFloat(e.target.value).toFixed(1);
        saveControlPanelState();
    });
}

if (weightVisualSlider && valWeightVisual) {
    weightVisualSlider.addEventListener("input", (e) => {
        valWeightVisual.innerText = parseFloat(e.target.value).toFixed(1);
        saveControlPanelState();
    });
}

if (weightOcrSlider && valWeightOcr) {
    weightOcrSlider.addEventListener("input", (e) => {
        valWeightOcr.innerText = parseFloat(e.target.value).toFixed(1);
        saveControlPanelState();
    });
}

if (weightAsrSlider && valWeightAsr) {
    weightAsrSlider.addEventListener("input", (e) => {
        valWeightAsr.innerText = parseFloat(e.target.value).toFixed(1);
        saveControlPanelState();
    });
}

// Đồng bộ Visual Sim Slider & Number Input
if (visualSimSlider && visualSimNum) {
    visualSimSlider.addEventListener("input", (e) => {
        visualSimNum.value = parseFloat(e.target.value).toFixed(3);
        saveControlPanelState();
    });

    visualSimNum.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value) || 0.90;
        visualSimSlider.value = val;
        saveControlPanelState();
    });
}

// Đồng bộ Cross-Video Dedup Slider & Number Input
if (crossVideoSimSlider && crossVideoSimNum) {
    crossVideoSimSlider.addEventListener("input", (e) => {
        crossVideoSimNum.value = parseFloat(e.target.value).toFixed(3);
        saveControlPanelState();
    });

    crossVideoSimNum.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value) || 0.90;
        crossVideoSimSlider.value = val;
        saveControlPanelState();
    });
}

// Đồng bộ TRAKE Candidate Pool Slider & Number Input & Tự động lưu
if (trakeCandidatePoolSlider && trakeCandidatePoolInput) {
    trakeCandidatePoolSlider.addEventListener("input", (e) => {
        trakeCandidatePoolInput.value = parseInt(e.target.value, 10);
        saveControlPanelState();
    });

    trakeCandidatePoolInput.addEventListener("input", (e) => {
        const val = parseInt(e.target.value, 10) || 100;
        trakeCandidatePoolSlider.value = val;
        saveControlPanelState();
    });
}

// Gắn auto-save cho tất cả các input tham số TRAKE TARS DP
[trakeTopVInput, trakeFramesPerVidInput, trakeLambdaInput, trakeKpathsInput, trakeTopKInput].forEach(inp => {
    if (inp) {
        inp.addEventListener("input", saveControlPanelState);
        inp.addEventListener("change", saveControlPanelState);
    }
});

if (enableCrossVideoDedupCheck) {
    enableCrossVideoDedupCheck.addEventListener("change", () => {
        saveControlPanelState();
    });
}

// Đồng bộ Candidate Slider và Ô Nhập Số
if (topKSlider && topKNum) {
    topKSlider.addEventListener("input", (e) => {
        topKNum.value = e.target.value;
    });

    topKNum.addEventListener("input", (e) => {
        const val = parseInt(e.target.value) || 1;
        if (val <= 100) {
            topKSlider.value = val;
        } else {
            topKSlider.value = 100;
        }
    });
}

// Kẹp điều kiện giữa Smart Dedup và Unique Video
if (smartDedupCheck && uniqueVideoCheck) {
    uniqueVideoCheck.addEventListener("change", (e) => {
        if (e.target.checked) {
            smartDedupCheck.checked = false;
            bodyDedup.classList.add("disabled-body");
        }
    });

    smartDedupCheck.addEventListener("change", (e) => {
        if (e.target.checked) {
            uniqueVideoCheck.checked = false;
        }
    });
}

// Load & Save Gemini API Key vào localStorage. Không nhúng secret vào source.
if (geminiApiKeyInput) {
    const savedKey = localStorage.getItem("gemini_api_key");
    geminiApiKeyInput.value = savedKey || "";

    geminiApiKeyInput.addEventListener("input", (e) => {
        localStorage.setItem("gemini_api_key", e.target.value.trim());
    });
}

// ==========================================================================
// ASR EXPANDED CONTEXT PILLS (±N SEGMENTS)
// ==========================================================================
if (asrWindowPills) {
    asrWindowPills.querySelectorAll(".btn-pill").forEach((btn) => {
        btn.addEventListener("click", () => {
            asrWindowPills.querySelectorAll(".btn-pill").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedAsrWindowN = parseInt(btn.getAttribute("data-n")) || 0;
            saveControlPanelState();
            showToast(`🎙️ Đã chọn ASR Ngữ Cảnh mở rộng: ±${selectedAsrWindowN} cụm thoại liền kề`, "info", 1500);
        });
    });
}

// ==========================================================================
// DISCRIMINATING FEATURES TAGS HANDLER (QWEN GUIDE)
// ==========================================================================
function renderDiscriminatingTags() {
    if (!discriminatingTagsContainer) return;
    discriminatingTagsContainer.innerHTML = "";
    currentDiscriminatingFeatures.forEach((feat, idx) => {
        const tag = document.createElement("span");
        tag.className = "discriminating-tag";
        tag.innerHTML = `<span>${feat}</span><button type="button" class="btn-remove-tag" data-idx="${idx}" title="Xóa tag">&times;</button>`;
        tag.querySelector(".btn-remove-tag").addEventListener("click", () => {
            currentDiscriminatingFeatures.splice(idx, 1);
            renderDiscriminatingTags();
        });
        discriminatingTagsContainer.appendChild(tag);
    });
}

if (btnAddDiscFeature && addDiscFeatureInput) {
    const handleAddTag = () => {
        const val = addDiscFeatureInput.value.trim();
        if (val && !currentDiscriminatingFeatures.includes(val)) {
            currentDiscriminatingFeatures.push(val);
            renderDiscriminatingTags();
            addDiscFeatureInput.value = "";
        }
    };
    btnAddDiscFeature.addEventListener("click", handleAddTag);
    addDiscFeatureInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleAddTag();
        }
    });
}

// ==========================================================================
// QWEN TOP K CHAINS PILLS HANDLER
// ==========================================================================
if (qwenTopkPills && qwenTopkChainsInput) {
    qwenTopkPills.querySelectorAll(".btn-pill").forEach((btn) => {
        btn.addEventListener("click", () => {
            qwenTopkPills.querySelectorAll(".btn-pill").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const kVal = parseInt(btn.getAttribute("data-k")) || 5;
            qwenTopkChainsInput.value = kVal;
        });
    });

    qwenTopkChainsInput.addEventListener("input", (e) => {
        const val = parseInt(e.target.value) || 5;
        qwenTopkPills.querySelectorAll(".btn-pill").forEach((btn) => {
            if (parseInt(btn.getAttribute("data-k")) === val) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    });
}

// ==========================================================================
// GEMINI PROMPT PARSER (TÔN TRỌNG TRẠNG THÁI CHECKBOX CỦA NGƯỜI DÙNG)
// ==========================================================================
function parseLocalTrakeEvents(rawText) {
    const normalized = String(rawText || "").replace(/\r\n?/g, "\n").trim();
    if (!normalized) return [];

    const explicitMatches = Array.from(normalized.matchAll(/E\d+\s*:\s*([\s\S]*?)(?=(?:\n?\s*E\d+\s*:)|$)/gi));
    let eventTexts = explicitMatches.map(match => match[1].trim()).filter(Boolean);
    if (eventTexts.length < 2) {
        eventTexts = normalized
            .split(/\n[\t ]*\n+/)
            .map(block => block.replace(/\s*\n\s*/g, " ").trim())
            .filter(Boolean);
    }
    if (eventTexts.length < 2) return [];

    return eventTexts.map((text, index) => ({
        scene_idx: index + 1,
        name: `E${index + 1}: ${text.slice(0, 36)}${text.length > 36 ? "…" : ""}`,
        visual_query: text,
        ocr_query: "",
        speech_query: "",
        object_filter: null
    }));
}

function reindexTrakeScenes() {
    latestParsedScenes.forEach((scene, index) => {
        const text = String(scene.visual_query || "").trim();
        scene.scene_idx = index + 1;
        scene.name = `E${index + 1}: ${text.slice(0, 36)}${text.length > 36 ? "…" : ""}`;
    });
}

function renderTrakeEventEditor(activeIndex = 0) {
    if (!multiSceneEditorCards || currentAppMode !== "TRAKE" || latestParsedScenes.length < 2) return;
    const safeIndex = Math.max(0, Math.min(activeIndex, latestParsedScenes.length - 1));
    const scene = latestParsedScenes[safeIndex];
    activeSceneTabIdx = safeIndex;
    multiSceneTabs?.querySelectorAll(".scene-tab-btn").forEach((button, index) => {
        button.classList.toggle("active-scene-tab", index === safeIndex + 1);
    });
    multiSceneEditorCards.style.display = "block";
    multiSceneEditorCards.innerHTML = "";

    const editor = document.createElement("div");
    editor.className = "trake-event-editor";

    const header = document.createElement("div");
    header.className = "trake-event-editor-header";
    const title = document.createElement("strong");
    title.textContent = `E${safeIndex + 1}`;
    const hint = document.createElement("span");
    hint.textContent = "Alt+↑/↓ đổi vị trí";
    header.append(title, hint);

    const textarea = document.createElement("textarea");
    textarea.rows = 2;
    textarea.value = scene.visual_query || "";
    textarea.setAttribute("aria-label", `Nội dung sự kiện E${safeIndex + 1}`);
    textarea.addEventListener("input", () => {
        scene.visual_query = textarea.value;
        reindexTrakeScenes();
        const tab = document.getElementById(`scene-tab-btn-${safeIndex}`);
        if (tab) tab.title = textarea.value;
        if (searchInput && activeSceneTabIdx === safeIndex) searchInput.value = textarea.value;
    });
    textarea.addEventListener("keydown", (event) => {
        if (!event.altKey || !["ArrowUp", "ArrowDown"].includes(event.key)) return;
        event.preventDefault();
        const direction = event.key === "ArrowUp" ? -1 : 1;
        const targetIndex = safeIndex + direction;
        if (targetIndex < 0 || targetIndex >= latestParsedScenes.length) return;
        [latestParsedScenes[safeIndex], latestParsedScenes[targetIndex]] = [latestParsedScenes[targetIndex], latestParsedScenes[safeIndex]];
        reindexTrakeScenes();
        renderMultiSceneTabs(latestParsedScenes);
        renderTrakeEventEditor(targetIndex);
    });

    const actions = document.createElement("div");
    actions.className = "trake-event-editor-actions";
    const makeAction = (label, titleText, handler, danger = false) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.title = titleText;
        if (danger) button.classList.add("danger");
        button.addEventListener("click", handler);
        return button;
    };
    const insertAt = index => {
        latestParsedScenes.splice(index, 0, { visual_query: "", ocr_query: "", speech_query: "", object_filter: null });
        reindexTrakeScenes();
        renderMultiSceneTabs(latestParsedScenes);
        renderTrakeEventEditor(index);
        multiSceneEditorCards.querySelector("textarea")?.focus();
    };
    actions.append(
        makeAction("＋ Trước", "Chèn event trước vị trí hiện tại", () => insertAt(safeIndex)),
        makeAction("＋ Sau", "Chèn event sau vị trí hiện tại", () => insertAt(safeIndex + 1)),
        makeAction("Xóa", "Xóa event hiện tại", () => {
            if (latestParsedScenes.length <= 2) {
                showToast("TRAKE cần ít nhất 2 sự kiện.", "warning", 2200);
                return;
            }
            latestParsedScenes.splice(safeIndex, 1);
            reindexTrakeScenes();
            const nextIndex = Math.min(safeIndex, latestParsedScenes.length - 1);
            renderMultiSceneTabs(latestParsedScenes);
            renderTrakeEventEditor(nextIndex);
        }, true)
    );

    editor.append(header, textarea, actions);
    multiSceneEditorCards.appendChild(editor);
}

function activateLocalTrakeDraft(scenes, sourceInput = null) {
    latestParsedScenes = scenes;
    latestUnifiedVisual = scenes.map(scene => scene.visual_query).join("; ");
    latestRerankerQuery = sourceInput?.value.trim() || rawQueryInput?.value.trim() || latestUnifiedVisual;
    isMultiSceneMode = true;
    switchMode("TRAKE");
    renderMultiSceneTabs(scenes);
    renderTrakeEventEditor(0);
    if (multiSceneBar) multiSceneBar.style.display = "flex";
    if (multiSceneBadgeTitle) multiSceneBadgeTitle.innerText = `⏱️ TRAKE (${scenes.length} EVENTS)`;
    if (multiSceneSummary) multiSceneSummary.innerText = `${scenes.length} sự kiện · Enter lần nữa để tìm`;
    // Giữ nguyên văn bản nhiều đoạn nếu Visual Query là nơi nhập nguồn. Nếu thay
    // bằng E1 ở đây thì chữ ký draft đổi và Enter lần hai không thể chạy TRAKE.
    if (searchInput && sourceInput !== searchInput) searchInput.value = scenes[0].visual_query;
    statusText.innerText = `Đã định dạng E1–E${scenes.length}. Kiểm tra/chỉnh sửa rồi nhấn Enter lần nữa để SEARCH TRAKE.`;
    showToast(`Đã nhận diện ${scenes.length} sự kiện TRAKE. Enter lần nữa để tìm.`, "success", 3000);
}

function resetTrakeEnterState(sourceInput = null) {
    if (!sourceInput || trakeDraftSourceId === sourceInput.id) {
        trakeEnterStage = 0;
        trakeDraftSignature = "";
        trakeDraftSourceId = "";
    }
}

function handleTrakeDraftEnter(event, sourceInput) {
    if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return;

    const signature = sourceInput.value.replace(/\r\n?/g, "\n").trim();
    if (
        trakeEnterStage === 1
        && trakeDraftSourceId === sourceInput.id
        && signature === trakeDraftSignature
        && latestParsedScenes.length >= 2
    ) {
        event.preventDefault();
        event.stopPropagation();
        resetTrakeEnterState();
        performTrakeSearch();
        return;
    }

    const scenes = parseLocalTrakeEvents(signature);
    if (scenes.length >= 2) {
        event.preventDefault();
        event.stopPropagation();
        activateLocalTrakeDraft(scenes, sourceInput);
        trakeEnterStage = 1;
        trakeDraftSignature = signature;
        trakeDraftSourceId = sourceInput.id;
        return;
    }

    // Visual Query một cảnh vẫn giữ hành vi Enter = tìm kiếm thông thường.
    if (sourceInput === searchInput) {
        event.preventDefault();
        event.stopPropagation();
        performSearch();
    }
}

[rawQueryInput, searchInput].forEach(sourceInput => {
    if (!sourceInput) return;
    sourceInput.addEventListener("input", () => resetTrakeEnterState(sourceInput));
    sourceInput.addEventListener("keydown", event => handleTrakeDraftEnter(event, sourceInput));
});

async function parseWithGemini() {
    const rawText = rawQueryInput ? rawQueryInput.value.trim() : "";
    if (!rawText) {
        statusText.innerText = "⚠️ Vui lòng dán đề bài thô vào ô trên trước.";
        showToast("⚠️ Vui lòng dán đề bài thô vào ô trên trước.", "warning");
        return;
    }

    const apiKey = geminiApiKeyInput ? geminiApiKeyInput.value.trim() : "";

    aiParseBtn.disabled = true;
    aiParseBtn.innerText = "⏳ ĐANG PHÂN TÁCH...";
    statusText.innerText = "Đang gửi đề bài tới Gemini AI để bóc tách...";
    showToast("⏳ Đang gửi đề bài tới Gemini AI để bóc tách...", "loading", 0);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    try {
        const response = await fetch(PARSE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ raw_text: rawText, api_key: apiKey }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Parse Error: ${response.statusText}`);
        }

        const resData = await response.json();
        dismissLoadingToast();
        const data = resData.data;

        // Lưu lại câu reranker_query chi tiết và các câu thống nhất
        latestRerankerQuery = data.reranker_query || rawText;
        latestUnifiedVisual = data.visual_query || "";
        latestUnifiedSpeech = data.speech_query || "";
        latestUnifiedOcr = data.ocr_query || "";

        // Điền tự động nội dung chữ vào các ô tìm kiếm
        if (searchInput && data.visual_query) searchInput.value = data.visual_query;
        if (ocrInput) ocrInput.value = data.ocr_query || "";
        if (speechInput) speechInput.value = data.speech_query || "";

        // Nạp và hiển thị các đặc trưng phân biệt & gợi ý AI Rerank
        currentDiscriminatingFeatures = Array.isArray(data.discriminating_features) ? [...data.discriminating_features] : [];
        renderDiscriminatingTags();
        if (rerankHintInput) {
            rerankHintInput.value = data.rerank_hint || "";
        }
        if (geminiRerankGuidance) {
            geminiRerankGuidance.style.display = RERANK_FEATURE_VISIBLE && (data.task_type === "TRAKE" || data.is_multi_scene || currentDiscriminatingFeatures.length > 0) ? "block" : "none";
        }

        // Điền gợi ý lớp vật thể RT-DETR (KHÔNG tự động bật công tắc)
        if (data.object_filter && data.object_filter.class_id !== undefined && data.object_filter.class_id >= 0) {
            if (objectClassSelect) {
                objectClassSelect.value = data.object_filter.class_id;
            }
            if (data.object_filter.min_confidence && objectConfSlider && valObjectConf) {
                objectConfSlider.value = data.object_filter.min_confidence;
                valObjectConf.innerText = parseFloat(data.object_filter.min_confidence).toFixed(2);
            }
        }

        // Hiển thị banner nhắc nhở câu hỏi QA (nếu có)
        if (qaCallout && qaQuestionText) {
            if (data.task_type === "QA" && data.qa_question) {
                qaQuestionText.innerText = data.qa_question;
                qaCallout.style.display = "flex";
            } else {
                qaCallout.style.display = "none";
            }
        }

        const parserSrc = data.source || "Gemini AI";

        // Xử lý bài toán TRAKE (Temporal Reasoning & Keyframe Extraction)
        if (data.task_type === "TRAKE" && data.scenes && data.scenes.length >= 2) {
            isMultiSceneMode = true;
            latestParsedScenes = data.scenes;
            renderMultiSceneTabs(data.scenes);
            if (multiSceneBar) multiSceneBar.style.display = "flex";
            if (multiSceneBadgeTitle) multiSceneBadgeTitle.innerText = `⏱️ TRAKE (${data.scenes.length} EVENTS)`;
            if (multiSceneSummary) {
                multiSceneSummary.innerText = `Chuỗi sự kiện: ${data.scenes.map(s => s.name || `E${s.scene_idx}`).join(" ➔ ")}`;
            }

            // Tự động chuyển sang TRAKE Mode
            switchMode("TRAKE");
            renderMultiSceneTabs(data.scenes);
            renderTrakeEventEditor(0);
            statusText.innerText = `⏱️ [TRAKE MODE - ${data.scenes.length} SỰ KIỆN - ${parserSrc}] Đã nạp bối cảnh và chuỗi E1..E${data.scenes.length}. Bấm '⏱️ SEARCH TRAKE' để chạy TARS Monotonic DP.`;
            showToast(`⏱️ Đã phát hiện bài toán TRAKE (${data.scenes.length} sự kiện tuần tự) -> Tự động chuyển sang TRAKE Mode!`, "success", 3000);

        } else if (data.is_multi_scene && data.scenes && data.scenes.length >= 2) {
            // Xử lý chế độ Đa Phân Cảnh KIS thông thường
            if (currentAppMode === "TRAKE") switchMode("KIS");
            isMultiSceneMode = true;
            latestParsedScenes = data.scenes;
            renderMultiSceneTabs(data.scenes);
            if (multiSceneBar) multiSceneBar.style.display = "flex";
            if (multiSceneBadgeTitle) multiSceneBadgeTitle.innerText = "🎬 MULTI-SCENE";
            if (multiSceneSummary) {
                multiSceneSummary.innerText = `Phát hiện ${data.scenes.length} phân cảnh nối tiếp: ${data.scenes.map(s => s.name || `Cảnh ${s.scene_idx}`).join(" ➔ ")}`;
            }
            if (multiSearchBtn) {
                multiSearchBtn.disabled = !backendReady;
                multiSearchBtn.classList.add("active-multi-ready");
            }
            statusText.innerText = `✨ [ĐA PHÂN CẢNH ${data.scenes.length} ĐOẠN - ${parserSrc}] Nút 🎬 MULTI-SCENE đã mở khóa. Bấm '🚀 SEARCH NOW' (Đơn Cảnh) hoặc '🎬 MULTI-SCENE' (Giao Thoa).`;
            showToast(`✨ Đã mở khóa nút 🎬 MULTI-SCENE (${data.scenes.length} cảnh)!`, "success");
        } else {
            // Đơn cảnh thông thường
            if (currentAppMode === "TRAKE") switchMode("KIS");
            isMultiSceneMode = false;
            latestParsedScenes = [];
            if (multiSceneBar) multiSceneBar.style.display = "none";
            if (multiSearchBtn) {
                multiSearchBtn.disabled = true;
                multiSearchBtn.classList.remove("active-multi-ready");
            }
            statusText.innerText = `✨ [${data.task_type} - ${parserSrc}] Đã nạp nội dung tự nhiên. Bấm '🚀 SEARCH NOW'.`;
            showToast(`✨ Đã nạp nội dung phân tách (${parserSrc})!`, "success");
        }

    } catch (err) {
        clearTimeout(timeoutId);
        dismissLoadingToast();
        const errMsg = err.name === "AbortError" ? "Quá thời gian kết nối (Timeout)" : err.message;
        statusText.innerText = `[ERROR AI] ${errMsg}`;
        showToast(`Lỗi AI Parser: ${errMsg}`, "error");
        console.error(err);
    } finally {
        aiParseBtn.disabled = false;
        aiParseBtn.innerText = "✨ PHÂN TÁCH ĐỀ BÀI";
    }
}

if (aiParseBtn) {
    aiParseBtn.addEventListener("click", parseWithGemini);
}

// Toggle ẩn hiện Bảng Soạn Thảo Cảnh
if (toggleSceneEditorBtn && multiSceneEditorCards) {
    toggleSceneEditorBtn.addEventListener("click", () => {
        if (multiSceneEditorCards.style.display === "none") {
            multiSceneEditorCards.style.display = "grid";
            toggleSceneEditorBtn.innerText = "👁️ Ẩn Bảng Soạn Thảo";
        } else {
            multiSceneEditorCards.style.display = "none";
            toggleSceneEditorBtn.innerText = "✏️ Hiện Bảng Soạn Thảo";
        }
    });
}

// ==========================================================================
// RENDER THANH SEGMENTED PILLS ĐA PHÂN CẢNH (MULTI-SCENE SWITCHER)
// ==========================================================================
function renderMultiSceneTabs(scenes) {
    if (!multiSceneTabs) return;
    multiSceneTabs.innerHTML = "";

    // 1. Pill "Đầy Đủ (Đơn Cảnh)"
    const allTab = document.createElement("button");
    allTab.className = "scene-tab-btn active-scene-tab";
    allTab.innerText = currentAppMode === "TRAKE" ? "Chuỗi" : "🎬 Đầy Đủ (Đơn Cảnh)";
    allTab.title = "Truy vấn toàn bộ đề bài thành 1 cảnh đơn";
    allTab.addEventListener("click", () => {
        switchSceneTab(-1);
    });
    multiSceneTabs.appendChild(allTab);

    // 2. Pills từng phân cảnh con
    scenes.forEach((sc, idx) => {
        const tabBtn = document.createElement("button");
        tabBtn.className = "scene-tab-btn";
        tabBtn.id = `scene-tab-btn-${idx}`;
        const titleText = sc.name || `Cảnh ${idx + 1}`;
        tabBtn.innerText = currentAppMode === "TRAKE" ? `E${idx + 1}` : `${idx + 1}️⃣ ${titleText}`;
        tabBtn.title = sc.visual_query || "";
        tabBtn.addEventListener("click", () => {
            switchSceneTab(idx);
        });
        multiSceneTabs.appendChild(tabBtn);
    });
}

// ==========================================================================
// RENDER MULTI-SCENE STORYBOARD (GOM CỤM THEO HÀNG VIDEO TUẦN TỰ THỜI GIAN)
// ==========================================================================
function renderMultiSceneStoryboard(storyboardSequences) {
    if (!resultsGrid) return;
    resultsGrid.classList.add("trake-mode-grid");
    resultsGrid.innerHTML = "";

    if (!storyboardSequences || storyboardSequences.length === 0) {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 40px; text-align: center; color: var(--text-dim);">Không tìm thấy chuỗi phân cảnh nào phù hợp.</div>`;
        return;
    }

    const container = document.createElement("div");
    container.className = "storyboard-container";

    storyboardSequences.forEach((seq) => {
        const groupCard = document.createElement("div");
        groupCard.className = "storyboard-group-card";

        const chronoBadge = seq.is_chronological
            ? `<span class="storyboard-chrono-badge">⏱️ Chuẩn Tuần Tự (t₁ < t₂ < ...)</span>`
            : `<span class="storyboard-chrono-badge" style="background: #854d0e; color: #fef08a;">⚠️ Lệch Thứ Tự</span>`;

        let scenesHtml = "";
        seq.scenes.forEach((sc, scIdx) => {
            const timeSec = sc.timestamp_sec !== undefined ? Number(sc.timestamp_sec) : 0;
            const mins = Math.floor(timeSec / 60);
            const secs = Math.floor(timeSec % 60);
            const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            const scoreVal = sc.score !== undefined ? Number(sc.score).toFixed(4) : "0.0000";

            let arrowHtml = "";
            if (sc.delta_t_to_next !== undefined && sc.delta_t_to_next !== null) {
                const deltaSec = sc.delta_t_to_next;
                const deltaM = Math.floor(deltaSec / 60);
                const deltaS = Math.floor(deltaSec % 60);
                const deltaStr = deltaM > 0 ? `+${deltaM}m ${deltaS}s` : `+${deltaS}s`;

                arrowHtml = `
                    <div class="storyboard-arrow-wrapper">
                        <span class="storyboard-arrow-icon">➔</span>
                        <span class="storyboard-delta-time">${deltaStr}</span>
                    </div>
                `;
            }

            scenesHtml += `
                <div class="storyboard-scene-card" data-vid="${sc.video_id}" data-fid="${sc.frame_id}">
                    <div class="storyboard-scene-tag">🎬 ${sc.scene_name || `Cảnh ${sc.scene_idx}`}</div>
                    <div class="storyboard-thumb-wrapper" title="🔍 Bấm để soi Bounding Box">
                        <img src="${sc.image_url}" loading="lazy" alt="Frame ${sc.frame_id}">
                        <span class="card-score-pill">${scoreVal}</span>
                    </div>
                    <div class="storyboard-meta-bar">
                        <div class="storyboard-info-row">
                            <span style="font-weight: 700; color: #38bdf8;">⏱️ ${timeStr} (${timeSec}s)</span>
                            <span style="color: var(--text-dim);">F:${sc.frame_id}</span>
                        </div>
                        <div class="storyboard-actions-row">
                            <button class="btn-card-action btn-sb-video" title="Xem video tại giây này">▶️ Video</button>
                            <button class="btn-card-action btn-sb-copy" title="Copy Video ID & Frame ID">📋 Copy ID</button>
                        </div>
                    </div>
                </div>
                ${arrowHtml}
            `;
        });

        groupCard.innerHTML = `
            <div class="storyboard-group-header">
                <div class="storyboard-header-left">
                    <span class="storyboard-vid-badge">🏆 #${seq.rank} Video: ${seq.video_id}</span>
                    <span class="storyboard-joint-score">Score Chuỗi: ${seq.joint_score}</span>
                    ${chronoBadge}
                </div>
                <div class="storyboard-header-actions">
                    <button class="btn-card-action btn-sb-timeline-all" data-vid="${seq.video_id}">🎞️ Xem Timeline Toàn Bộ Video</button>
                </div>
            </div>
            <div class="storyboard-sequence-row">
                ${scenesHtml}
            </div>
        `;

        // Gắn sự kiện tương tác cho từng scene card
        groupCard.querySelectorAll(".storyboard-scene-card").forEach((scCard, idx) => {
            const scData = seq.scenes[idx];
            const imgBox = scCard.querySelector(".storyboard-thumb-wrapper");
            if (imgBox) {
                imgBox.addEventListener("click", () => {
                    openLightbox(0, [scData]);
                });
            }

            const vidBtn = scCard.querySelector(".btn-sb-video");
            if (vidBtn) {
                vidBtn.addEventListener("click", () => {
                    openVideoPlayer(scData.video_id, scData.frame_id, scData.timestamp_sec);
                });
            }

            const copyBtn = scCard.querySelector(".btn-sb-copy");
            if (copyBtn) {
                copyBtn.addEventListener("click", () => {
                    const text = `${scData.video_id} ${scData.frame_id}`;
                    navigator.clipboard.writeText(text).then(() => {
                        copyBtn.innerText = "✅ ĐÃ COPY!";
                        showToast(`📋 Đã sao chép: ${text}`, "success", 1800);
                        setTimeout(() => { copyBtn.innerText = "📋 Copy ID"; }, 1500);
                    });
                });
            }
        });

        // Nút xem toàn bộ timeline của video
        const tlAllBtn = groupCard.querySelector(".btn-sb-timeline-all");
        if (tlAllBtn) {
            tlAllBtn.addEventListener("click", () => {
                const firstFrame = seq.scenes[0] ? seq.scenes[0].frame_id : null;
                openTimeline(seq.video_id, firstFrame);
            });
        }

        container.appendChild(groupCard);
    });

    resultsGrid.appendChild(container);
}

function switchSceneTab(tabIdx) {
    activeSceneTabIdx = tabIdx;
    if (!multiSceneTabs) return;

    // Cập nhật trạng thái active của các nút Tab
    const buttons = multiSceneTabs.querySelectorAll(".scene-tab-btn");
    buttons.forEach((btn, idx) => {
        if (idx === tabIdx + 1) {
            btn.classList.add("active-scene-tab");
        } else {
            btn.classList.remove("active-scene-tab");
        }
    });

    // Đồng bộ nội dung sang Sidebar để người dùng dễ dàng xem và chỉnh sửa
    if (tabIdx >= 0 && latestParsedScenes && latestParsedScenes[tabIdx]) {
        const sc = latestParsedScenes[tabIdx];
        if (searchInput) searchInput.value = sc.visual_query || "";
        if (ocrInput) ocrInput.value = sc.ocr_query || "";
        if (speechInput) speechInput.value = sc.speech_query || "";
        if (statusText) statusText.innerText = `🎬 Đang xem [Cảnh ${tabIdx + 1}: ${sc.name || ""}]. Bấm "🚀 SEARCH NOW" (cho riêng cảnh này) hoặc "🎬 MULTI-SCENE" (quét tất cả).`;
    } else if (tabIdx === -1) {
        if (searchInput) searchInput.value = latestUnifiedVisual || latestRerankerQuery || "";
        if (ocrInput) ocrInput.value = latestUnifiedOcr || "";
        if (speechInput) speechInput.value = latestUnifiedSpeech || "";
        if (statusText) statusText.innerText = "🎬 Đang ở chế độ Đầy Đủ (Giao Thoa Storyboard). Bấm '🚀 SEARCH NOW' để tìm đơn cảnh, hoặc '🎬 MULTI-SCENE' để quét giao thoa.";
    }

    if (currentAppMode === "TRAKE" && tabIdx >= 0) {
        renderTrakeEventEditor(tabIdx);
    }

    // Nếu đã có kết quả tìm kiếm trước đó thì chuyển đổi chế độ xem kết quả
    if (!multiSceneResultsData) return;

    if (tabIdx === -1) {
        currentResults = multiSceneResultsData.combined_results || [];
        activeResultsSource = currentResults;
        resultCount.innerText = `${currentResults.length} items (Giao Thoa Đa Cảnh)`;
        renderResults(currentResults);
    } else {
        const scResults = multiSceneResultsData.scenes_results[tabIdx]?.results || [];
        currentResults = scResults;
        activeResultsSource = currentResults;
        const scName = multiSceneResultsData.scenes_results[tabIdx]?.name || `Cảnh ${tabIdx + 1}`;
        resultCount.innerText = `${currentResults.length} items (${scName})`;
        renderResults(currentResults);
    }
}

// Đồng bộ 2 chiều: Khi người dùng gõ trực tiếp vào sidebar inputs trong lúc chọn 1 Cảnh con
if (searchInput) {
    searchInput.addEventListener("input", (e) => {
        if (activeSceneTabIdx >= 0 && latestParsedScenes && latestParsedScenes[activeSceneTabIdx]) {
            latestParsedScenes[activeSceneTabIdx].visual_query = e.target.value;
        }
    });
}
if (ocrInput) {
    ocrInput.addEventListener("input", (e) => {
        if (activeSceneTabIdx >= 0 && latestParsedScenes && latestParsedScenes[activeSceneTabIdx]) {
            latestParsedScenes[activeSceneTabIdx].ocr_query = e.target.value;
        }
    });
}
if (speechInput) {
    speechInput.addEventListener("input", (e) => {
        if (activeSceneTabIdx >= 0 && latestParsedScenes && latestParsedScenes[activeSceneTabIdx]) {
            latestParsedScenes[activeSceneTabIdx].speech_query = e.target.value;
        }
    });
}

// ==========================================================================
// HÀM TRUY VẤN ĐA PHÂN CẢNH (MULTI-SCENE SEARCH & VIDEO INTERSECTION)
// ==========================================================================
async function performMultiSceneSearch() {
    if (!latestParsedScenes || latestParsedScenes.length < 2) return;

    const isVis = enableVisualCheck ? enableVisualCheck.checked : true;
    const isOcr = enableOcrCheck ? enableOcrCheck.checked : true;
    const isAsr = enableAsrCheck ? enableAsrCheck.checked : true;
    const isObj = false;

    let dedupMode = "none";
    if (uniqueVideoCheck && uniqueVideoCheck.checked) {
        dedupMode = "unique";
    } else if (smartDedupCheck && smartDedupCheck.checked) {
        const selectedRadio = document.querySelector('input[name="smart-submode"]:checked');
        dedupMode = selectedRadio ? selectedRadio.value : "smart";
    }

    const topK1Val = parseInt(topKNum ? topKNum.value : 100) || 100;
    const topNVal = parseInt(rerankTopV ? rerankTopV.value : 5) || 5;
    const mFramesVal = parseInt(rerankKPerVideo ? rerankKPerVideo.value : 40) || 40;

    const objClass = isObj ? parseInt(objectClassSelect ? objectClassSelect.value : 0) : null;
    const objConf = isObj ? parseFloat(objectConfSlider ? objectConfSlider.value : 0.3) : 0.0;
    const simThresh = parseFloat(visualSimNum ? visualSimNum.value : 0.90) || 0.90;

    const wVisual = parseFloat(weightVisualSlider ? weightVisualSlider.value : 0.5);
    const wOcr = parseFloat(weightOcrSlider ? weightOcrSlider.value : 0.2);
    const wAsr = parseFloat(weightAsrSlider ? weightAsrSlider.value : 0.2);
    const wObj = parseFloat(weightObjectSlider ? weightObjectSlider.value : 0.1);

    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const rrfK = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;
    const taxonomy = getTaxonomyFilterPayload();

    searchBtn.disabled = true;
    rerankBtn.disabled = true;
    if (multiSearchBtn) {
        multiSearchBtn.disabled = true;
        multiSearchBtn.innerText = "⏳ ĐANG QUÉT...";
    }
    statusText.innerText = `🎬 Đang quét đồng thời ${latestParsedScenes.length} phân cảnh & Giao thoa Top ${topNVal} video...`;
    showToast(`🎬 Đang quét ${latestParsedScenes.length} phân cảnh & Giao thoa Top ${topNVal} Video...`, "loading", 0);

    const t0 = performance.now();

    try {
        const response = await fetch(MULTI_SCENE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenes: latestParsedScenes,
                enable_visual: isVis,
                enable_ocr: isOcr,
                enable_asr: isAsr,
                enable_object: isObj,
                object_class_id: objClass,
                object_conf_thresh: objConf,
                weight_visual: wVisual,
                weight_ocr: wOcr,
                weight_asr: wAsr,
                weight_object: wObj,
                dedup_mode: dedupMode,
                visual_sim_thresh: simThresh,
                enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
                cross_video_sim_thresh: parseFloat(crossVideoSimNum ? crossVideoSimNum.value : 0.90) || 0.90,
                top_k: topK1Val,
                top_v: topNVal,
                k_per_video: mFramesVal,
                use_rrf: isRrf,
                rrf_k: rrfK,
                asr_window_n: selectedAsrWindowN,
                rrf_use_visual: rrfSignalVisual ? rrfSignalVisual.checked : true,
                rrf_use_asr: rrfSignalAsr ? rrfSignalAsr.checked : true,
                rrf_use_shot: rrfSignalShot ? rrfSignalShot.checked : true,
                rrf_use_caption: rrfSignalCaption ? rrfSignalCaption.checked : false,
                category: taxonomy.categories,
                sub_category: taxonomy.subjects,
                include_videos: currentIncludeTags.length > 0 ? currentIncludeTags : null,
                exclude_videos: currentExcludeTags.length > 0 ? currentExcludeTags : null,
                return_shots: false
            })
        });

        if (!response.ok) {
            throw new Error(`Multi-Scene Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();
        multiSceneResultsData = data;
        currentResults = data.combined_results || [];
        const elapsedSec = data.elapsed_sec || ((performance.now() - t0) / 1000).toFixed(2);
        const topVList = (data.top_videos || []).join(", ");
        resultCount.innerText = `${currentResults.length} items (Giao Thoa ${latestParsedScenes.length} Cảnh)`;
        statusText.innerText = `🎯 ĐA PHÂN CẢNH: Giao thoa Top ${topNVal} Video [${topVList}] -> ${currentResults.length} frames (${elapsedSec}s)!`;
        showToast(`✅ Đã tìm thấy ${currentResults.length} frames từ Top ${topNVal} video (${elapsedSec}s)!`, "success");

        switchSceneTab(-1);

    } catch (err) {
        dismissLoadingToast();
        statusText.innerText = `[ERROR MULTI-SCENE] ${err.message}`;
        showToast(`Lỗi Multi-Scene: ${err.message}`, "error");
        console.error(err);
    } finally {
        searchBtn.disabled = !backendReady;
        rerankBtn.disabled = !backendReady;
        if (multiSearchBtn) {
            multiSearchBtn.disabled = !backendReady || !isMultiSceneMode;
            multiSearchBtn.innerText = "🎬 MULTI-SCENE";
        }
        updateK2Calculation();
    }
}

// ==========================================================================
// RENDER TRAKE STORYBOARD MATRIX (TARS MONOTONIC DP RESULTS)
// ==========================================================================
function renderTrakeStoryboard(storyboardSequences) {
    if (!resultsGrid) return;
    resultsGrid.classList.add("trake-mode-grid");
    resultsGrid.innerHTML = "";

    // Hiển thị thanh công cụ Qwen AI Reranker khi ở chế độ TRAKE
    if (trakeQwenToolbar) {
        trakeQwenToolbar.style.display = RERANK_FEATURE_VISIBLE && (currentAppMode === "TRAKE" && storyboardSequences && storyboardSequences.length > 0) ? "flex" : "none";
    }

    if (!storyboardSequences || storyboardSequences.length === 0) {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 50px; text-align: center; color: var(--text-dim); font-size: 0.95rem;">
            ⏱️ Không tìm thấy chuỗi sự kiện TRAKE nào phù hợp. Vui lòng kiểm tra lại bối cảnh và các sự kiện con.
        </div>`;
        return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "trake-results-wrapper";

    storyboardSequences.forEach((seq) => {
        const groupCard = document.createElement("div");
        const vidUpper = (seq.video_id || "").toUpperCase();
        const isExcluded = currentExcludeTags.includes(vidUpper);
        groupCard.className = isExcluded ? "trake-group-card card-excluded" : "trake-group-card";
        groupCard.dataset.videoId = seq.video_id || "";

        const chronoBadge = seq.is_chronological
            ? `<span class="storyboard-chrono-badge">⏱️ Chuẩn Tuần Tự (t₁ < t₂ < ...)</span>`
            : `<span class="storyboard-chrono-badge" style="background: #854d0e; color: #fef08a;">⚠️ Lệch Thứ Tự</span>`;

        let qwenBadgeHtml = "";
        if (seq.qwen_score !== undefined && seq.qwen_score !== null) {
            let qwenClass = "qwen-high";
            if (seq.qwen_score < 0.5) qwenClass = "qwen-low";
            else if (seq.qwen_score < 0.8) qwenClass = "qwen-med";
            const deltaStr = seq.original_rank ? `<span class="storyboard-rank-delta">(TARS #${seq.original_rank})</span>` : "";
            qwenBadgeHtml = `<span class="storyboard-qwen-score ${qwenClass}">🤖 Qwen: ${Number(seq.qwen_score).toFixed(4)}</span> ${deltaStr}`;
        }

        let scenesHtml = "";
        seq.scenes.forEach((sc, scIdx) => {
            const timeSec = sc.timestamp_sec !== undefined ? Number(sc.timestamp_sec) : 0;
            const mins = Math.floor(timeSec / 60);
            const secs = Math.floor(timeSec % 60);
            const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            const scoreVal = sc.score !== undefined ? Number(sc.score).toFixed(4) : "0.0000";

            let arrowHtml = "";
            if (sc.delta_t_to_next !== undefined && sc.delta_t_to_next !== null) {
                const deltaSec = sc.delta_t_to_next;
                const deltaM = Math.floor(deltaSec / 60);
                const deltaS = Math.floor(deltaSec % 60);
                const deltaStr = deltaM > 0 ? `+${deltaM}m ${deltaS}s` : `+${deltaS}s`;

                arrowHtml = `
                    <div class="storyboard-arrow-wrapper">
                        <span class="storyboard-arrow-icon">➔</span>
                        <span class="storyboard-delta-time">${deltaStr}</span>
                    </div>
                `;
            }

            const eventLabel = sc.scene_name ? (sc.scene_name.includes(":") ? sc.scene_name.split(":")[0] : `E${scIdx + 1}`) : `E${scIdx + 1}`;
            scenesHtml += `
                <div class="storyboard-scene-card" data-vid="${sc.video_id}" data-fid="${sc.frame_id}">
                    <span class="trake-event-tag">${eventLabel}</span>
                    <div class="storyboard-thumb-wrapper" title="🔍 Bấm để xem chi tiết / Bounding Box" style="position: relative; overflow: hidden; cursor: pointer; height: 130px; background: #000;">
                        <img src="${sc.image_url}" loading="lazy" alt="Frame ${sc.frame_id}" class="sb-thumb-img" style="width: 100%; height: 100%; object-fit: cover; display: block;">
                        <span class="card-score-pill">${scoreVal}</span>
                    </div>
                    <div class="storyboard-meta-bar">
                        <div class="storyboard-info-row" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                            <span style="font-weight: 700; color: #38bdf8; font-size: 0.72rem;">⏱️ ${timeStr}</span>
                            <span style="color: var(--text-dim); font-weight: 700; font-size: 0.70rem;">F:${sc.frame_id}</span>
                        </div>
                        <div class="storyboard-actions-row">
                            <button class="btn-card-action btn-sb-video" title="Xem video đầy đủ tại giây này">▶️ Video</button>
                            <button class="btn-card-action btn-sb-copy" title="Copy Frame ID">📋 Copy</button>
                        </div>
                    </div>
                </div>
                ${arrowHtml}
            `;
        });

        groupCard.innerHTML = `
            <div class="trake-group-header">
                <div class="storyboard-header-left" style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span class="trake-vid-badge">🏆 #${seq.rank} Video: ${seq.video_id}</span>
                    <span style="font-family: var(--font-mono); font-size: 0.76rem; color: #cbd5e1;">Path #${seq.path_idx || 1}</span>
                    <span class="storyboard-joint-score">Score: ${seq.joint_score}</span>
                    ${chronoBadge}
                    ${qwenBadgeHtml}
                </div>
                <div class="storyboard-header-actions" style="display: flex; align-items: center; gap: 8px;">
                    <button class="btn-card-action btn-dres-submit-trake" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; font-weight: 700; padding: 6px 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(16,185,129,0.35); border: none; cursor: pointer;" data-vid="${seq.video_id}" title="Nộp toàn bộ chuỗi sự kiện này lên DRES">🚀 NỘP DRES CHUỖI NÀY</button>
                    <code class="trake-sub-code">${seq.submission_line}</code>
                    <button class="btn-card-action btn-copy-trake-line" data-line="${seq.submission_line}">📋 Copy Line</button>
                    <button class="btn-card-action btn-sb-timeline-all" data-vid="${seq.video_id}">🎞️ Timeline</button>
                    <button class="btn-sb-exclude ${isExcluded ? 'is-excluded' : ''}" data-vid="${seq.video_id}" title="Loại trừ / hoàn tác loại trừ video này khỏi TRAKE">
                        ${isExcluded ? '🔴 Đã loại • Bỏ loại' : '🚫 Loại video'}
                    </button>
                </div>
            </div>
            <div class="storyboard-sequence-row">
                ${scenesHtml}
            </div>
        `;

        // Gắn sự kiện tương tác cho từng scene card
        groupCard.querySelectorAll(".storyboard-scene-card").forEach((scCard, idx) => {
            const scData = seq.scenes[idx];
            const imgBox = scCard.querySelector(".storyboard-thumb-wrapper");
            if (imgBox) {
                imgBox.addEventListener("click", () => {
                    openLightbox(0, [scData]);
                });
            }

            const vidBtn = scCard.querySelector(".btn-sb-video");
            if (vidBtn) {
                vidBtn.addEventListener("click", () => {
                    openVideoPlayer(scData.video_id, scData.frame_id, scData.timestamp_sec);
                });
            }

            const copyBtn = scCard.querySelector(".btn-sb-copy");
            if (copyBtn) {
                copyBtn.addEventListener("click", () => {
                    const text = `${scData.video_id} ${scData.frame_id}`;
                    navigator.clipboard.writeText(text).then(() => {
                        copyBtn.innerText = "✅ ĐÃ COPY!";
                        showToast(`📋 Đã sao chép: ${text}`, "success", 1800);
                        setTimeout(() => { copyBtn.innerText = "📋 Copy"; }, 1500);
                    });
                });
            }
        });

        // Nút Copy toàn bộ dòng TRAKE Line
        const btnDresTrake = groupCard.querySelector(".btn-dres-submit-trake");
        if (btnDresTrake) {
            btnDresTrake.addEventListener("click", () => {
                const frameIds = seq.scenes.map(s => s.frame_id);
                triggerDresTrakeSubmit(seq.video_id, frameIds, btnDresTrake);
            });
        }

        const copyLineBtn = groupCard.querySelector(".btn-copy-trake-line");
        if (copyLineBtn) {
            copyLineBtn.addEventListener("click", () => {
                const subLine = copyLineBtn.getAttribute("data-line");
                navigator.clipboard.writeText(subLine).then(() => {
                    copyLineBtn.innerText = "✅ ĐÃ COPY LINE!";
                    showToast(`📋 Đã copy TRAKE submission line:\n${subLine}`, "success", 2500);
                    setTimeout(() => { copyLineBtn.innerText = "📋 Copy TRAKE Line"; }, 2000);
                });
            });
        }

        // Nút xem toàn bộ timeline của video
        const tlAllBtn = groupCard.querySelector(".btn-sb-timeline-all");
        if (tlAllBtn) {
            tlAllBtn.addEventListener("click", () => {
                const firstFrame = seq.scenes[0] ? seq.scenes[0].frame_id : null;
                openTimeline(seq.video_id, firstFrame);
            });
        }

        // Nút Quick-Exclude cho TRAKE Video Group
        const excludeBtn = groupCard.querySelector(".btn-sb-exclude");
        if (excludeBtn) {
            excludeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                const vUpper = (seq.video_id || "").toUpperCase();
                const eIdx = currentExcludeTags.indexOf(vUpper);
                if (eIdx === -1) {
                    currentExcludeTags.push(vUpper);
                    showToast(`🚫 Đã loại trừ video ${seq.video_id} khỏi TRAKE`, "warning", 2000);
                } else {
                    currentExcludeTags.splice(eIdx, 1);
                    showToast(`✅ Đã bỏ loại trừ video ${seq.video_id}`, "success", 2000);
                }
                renderExcludeTags();
                applyExcludeStateToGrid();
            });
        }

        wrapper.appendChild(groupCard);
    });

    resultsGrid.appendChild(wrapper);
}

// ==========================================================================
// HÀM QWEN-VL RE-RANK CHO CÁC CHUỖI TRAKE STORYBOARD (GIAI ĐOẠN 2)
// ==========================================================================
async function performQwenRerankChains() {
    if (!latestTrakeSequences || latestTrakeSequences.length === 0) {
        statusText.innerText = "⚠️ Chưa có dữ liệu chuỗi TRAKE. Vui lòng bấm '⏱️ SEARCH TRAKE' trước.";
        showToast("⚠️ Chưa có kết quả TRAKE. Vui lòng bấm '⏱️ SEARCH TRAKE' trước.", "warning", 3000);
        return;
    }

    const topKVal = parseInt(qwenTopkChainsInput ? qwenTopkChainsInput.value : 5) || 5;
    const globalQ = searchInput ? searchInput.value.trim() : (latestUnifiedVisual || "");
    const hintVal = rerankHintInput ? rerankHintInput.value.trim() : "";

    if (trakeQwenRerankBtn) {
        trakeQwenRerankBtn.disabled = true;
        trakeQwenRerankBtn.innerText = "⏳ ĐANG AI RERANK...";
    }
    if (qwenProgressText) qwenProgressText.innerText = `Đang gửi ${topKVal} chuỗi lên Qwen-VL GPU...`;
    showToast(`🤖 Đang chấm điểm Top ${topKVal} chuỗi TRAKE qua Qwen-VL GPU...`, "loading", 0);

    const t0 = performance.now();

    try {
        const response = await fetch(QWEN_RERANK_CHAINS_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                chains: latestTrakeSequences,
                global_query: globalQ,
                discriminating_features: currentDiscriminatingFeatures,
                rerank_hint: hintVal,
                top_k_chains: topKVal,
                asr_window_n: selectedAsrWindowN
            })
        });

        if (!response.ok) {
            throw new Error(`Qwen Rerank Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();
        const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

        if (data.storyboard_sequences && data.storyboard_sequences.length > 0) {
            latestTrakeSequences = data.storyboard_sequences;
            renderTrakeStoryboard(latestTrakeSequences);
            if (qwenProgressText) qwenProgressText.innerText = `✅ Đã chấm ${data.total_evaluated} chuỗi (${elapsed}s)`;
            statusText.innerText = `🤖 QWEN RERANK HOÀN TẤT: Đã tái xếp hạng ${data.total_evaluated} chuỗi TRAKE trong ${elapsed}s!`;
            showToast(`✅ Qwen-VL đã tái xếp hạng thành công ${data.total_evaluated} chuỗi (${elapsed}s)!`, "success");
        }
    } catch (err) {
        dismissLoadingToast();
        if (qwenProgressText) qwenProgressText.innerText = `❌ Lỗi: ${err.message}`;
        statusText.innerText = `[ERROR QWEN RERANK] ${err.message}`;
        showToast(`Lỗi Qwen Rerank: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (trakeQwenRerankBtn) {
            trakeQwenRerankBtn.disabled = false;
            trakeQwenRerankBtn.innerText = "🤖 QWEN RERANK CHAINS";
        }
    }
}

// Gắn sự kiện kích hoạt Qwen Reranker
if (trakeQwenRerankBtn) {
    trakeQwenRerankBtn.addEventListener("click", performQwenRerankChains);
}

// ==========================================================================
// HÀM TRUY VẤN TRAKE (TARS MONOTONIC DP & EVENT EXTRACTION)
// ==========================================================================
async function performTrakeSearch() {
    let scenesToSearch = latestParsedScenes;
    if (!scenesToSearch || scenesToSearch.length < 2) {
        statusText.innerText = "⚠️ Chế độ TRAKE yêu cầu ít nhất 2 sự kiện (E1, E2...). Vui lòng dán đề bài và bấm '✨ PHÂN TÁCH ĐỀ BÀI' trước.";
        showToast("⚠️ Vui lòng dán đề bài và bấm '✨ PHÂN TÁCH ĐỀ BÀI' để bóc tách các sự kiện E1..EN.", "warning", 3500);
        return;
    }

    const isVis = enableVisualCheck ? enableVisualCheck.checked : true;
    const isOcr = enableOcrCheck ? enableOcrCheck.checked : true;
    const isAsr = enableAsrCheck ? enableAsrCheck.checked : true;
    const isObj = false;

    let dedupMode = "none";
    if (uniqueVideoCheck && uniqueVideoCheck.checked) {
        dedupMode = "unique";
    } else if (smartDedupCheck && smartDedupCheck.checked) {
        const selectedRadio = document.querySelector('input[name="smart-submode"]:checked');
        dedupMode = selectedRadio ? selectedRadio.value : "smart";
    }

    const lambdaVal = parseFloat(trakeLambdaInput ? trakeLambdaInput.value : 0.001) || 0.001;
    const kPathsVal = parseInt(trakeKpathsInput ? trakeKpathsInput.value : 3) || 3;
    const requestedTopKSeqVal = parseInt(trakeTopKInput ? trakeTopKInput.value : 10) || 10;
    const topK1Val = parseInt(trakeCandidatePoolInput ? trakeCandidatePoolInput.value : (topKNum ? topKNum.value : 100)) || 100;
    const topVVal = Math.trunc(getBoundedNumberInput(trakeTopVInput, 10));
    const mFramesVal = Math.trunc(getBoundedNumberInput(trakeFramesPerVidInput, 40));
    const refineCallCount = scenesToSearch.length * topVVal;
    const maxSequenceCapacity = topVVal * kPathsVal;
    const topKSeqVal = Math.min(requestedTopKSeqVal, maxSequenceCapacity);

    const objClass = isObj ? parseInt(objectClassSelect ? objectClassSelect.value : 0) : null;
    const objConf = isObj ? parseFloat(objectConfSlider ? objectConfSlider.value : 0.3) : 0.0;
    const simThresh = parseFloat(visualSimNum ? visualSimNum.value : 0.90) || 0.90;

    const wVisual = isVis ? parseFloat(weightVisualSlider ? weightVisualSlider.value : 1.0) : 0.0;
    const wOcr = isOcr ? parseFloat(weightOcrSlider ? weightOcrSlider.value : 1.0) : 0.0;
    const wAsr = isAsr ? parseFloat(weightAsrSlider ? weightAsrSlider.value : 1.0) : 0.0;
    const wObj = isObj ? parseFloat(weightObjectSlider ? weightObjectSlider.value : 0.5) : 0.0;

    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const rrfK = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;
    const taxonomy = getTaxonomyFilterPayload();

    if (trakeSearchBtn) {
        trakeSearchBtn.disabled = true;
        trakeSearchBtn.innerText = "⏳ ĐANG CHẠY TARS DP...";
    }
    if (requestedTopKSeqVal > maxSequenceCapacity) {
        showToast(
            `Top Sequences được giới hạn ${requestedTopKSeqVal} → ${topKSeqVal} (Top V ${topVVal} × K-paths ${kPathsVal}).`,
            "warning",
            4200
        );
    }
    if (refineCallCount > 100) {
        showToast(
            `⚠️ Cấu hình nặng: ${scenesToSearch.length} events × ${topVVal} videos = ${refineCallCount} lượt refine.`,
            "warning",
            5200
        );
    }
    statusText.innerText = `⏱️ Đang thực thi TARS Monotonic DP (Pool K1=${topK1Val}, Top V=${topVVal}, λ=${lambdaVal}, K-paths=${kPathsVal})...`;
    showToast(`⏱️ Đang chạy TARS Monotonic DP (Pool K1=${topK1Val}, Top V=${topVVal})...`, "loading", 0);

    const t0 = performance.now();

    try {
        const response = await fetch(MULTI_SCENE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenes: scenesToSearch,
                enable_visual: isVis,
                enable_ocr: isOcr,
                enable_asr: isAsr,
                enable_object: isObj,
                object_class_id: objClass,
                object_conf_thresh: objConf,
                weight_visual: wVisual,
                weight_ocr: wOcr,
                weight_asr: wAsr,
                weight_object: wObj,
                dedup_mode: dedupMode,
                visual_sim_thresh: simThresh,
                enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
                cross_video_sim_thresh: parseFloat(crossVideoSimNum ? crossVideoSimNum.value : 0.90) || 0.90,
                top_k: topK1Val,
                top_v: topVVal,
                k_per_video: mFramesVal,
                use_rrf: isRrf,
                rrf_k: rrfK,
                trake_mode: true,
                lambda_penalty: lambdaVal,
                alpha_fusion: 0.35,
                k_paths: kPathsVal,
                trake_top_k: topKSeqVal,
                asr_window_n: selectedAsrWindowN,
                rrf_use_visual: rrfSignalVisual ? rrfSignalVisual.checked : true,
                rrf_use_asr: rrfSignalAsr ? rrfSignalAsr.checked : true,
                rrf_use_shot: rrfSignalShot ? rrfSignalShot.checked : true,
                rrf_use_caption: rrfSignalCaption ? rrfSignalCaption.checked : false,
                category: taxonomy.categories,
                sub_category: taxonomy.subjects,
                include_videos: currentIncludeTags.length > 0 ? currentIncludeTags : null,
                exclude_videos: currentExcludeTags.length > 0 ? currentExcludeTags : null,
                return_shots: false,
                compact_response: true
            })
        });

        if (!response.ok) {
            throw new Error(`TRAKE Search Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();

        latestTrakeSequences = data.storyboard_sequences || [];
        latestTrakeSubmissionLines = data.trake_submission_lines || [];
        const elapsedSec = data.elapsed_sec || ((performance.now() - t0) / 1000).toFixed(2);

        resultCount.innerText = `${latestTrakeSequences.length} sequences (${data.top_videos ? data.top_videos.length : 0} videos)`;
        statusText.innerText = `⏱️ TRAKE: TARS DP hoàn tất trong ${elapsedSec}s! Tìm thấy ${latestTrakeSequences.length} chuỗi tuần tự từ Top ${data.top_videos ? data.top_videos.length : 0} Video.`;
        showToast(`✅ Đã tìm thấy ${latestTrakeSequences.length} chuỗi TRAKE tối ưu (${elapsedSec}s)!`, "success");

        renderTrakeStoryboard(latestTrakeSequences);

    } catch (err) {
        dismissLoadingToast();
        statusText.innerText = `[ERROR TRAKE] ${err.message}`;
        showToast(`Lỗi TRAKE Search: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (trakeSearchBtn) {
            trakeSearchBtn.disabled = !backendReady;
            trakeSearchBtn.innerText = "⏱️ SEARCH TRAKE";
        }
    }
}

// ==========================================================================
// HÀM XUẤT CSV NỘP BÀI TRAKE (EXPORT SUBMISSION CSV)
// ==========================================================================
function exportTrakeCsv() {
    if (!latestTrakeSubmissionLines || latestTrakeSubmissionLines.length === 0) {
        statusText.innerText = "⚠️ Chưa có dữ liệu chuỗi TRAKE để xuất CSV. Vui lòng bấm '⏱️ SEARCH TRAKE' trước.";
        showToast("⚠️ Chưa có kết quả TRAKE. Vui lòng bấm '⏱️ SEARCH TRAKE' trước.", "warning", 3000);
        return;
    }

    const csvContent = latestTrakeSubmissionLines.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `aic2026_trake_submission_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast(`📥 Đã xuất thành công ${latestTrakeSubmissionLines.length} dòng submission vào file CSV!`, "success", 3000);
}

// ==========================================================================
// HÀM TÌM KIẾM ĐƠN CẢNH / THEO INPUT HIỆN TẠI (SEARCH NOW)
// ==========================================================================
async function performSearch() {
    const isVis = enableVisualCheck ? enableVisualCheck.checked : true;
    const isOcr = enableOcrCheck ? enableOcrCheck.checked : true;
    const isAsr = enableAsrCheck ? enableAsrCheck.checked : true;
    const isObj = false;

    const qVisual = isVis && searchInput ? searchInput.value.trim() : "";
    const qOcr = isOcr && ocrInput ? ocrInput.value.trim() : "";
    const qSpeech = isAsr && speechInput ? speechInput.value.trim() : "";
    const vId = videoIdInput ? videoIdInput.value.trim() : "";

    let dedupMode = "none";
    if (uniqueVideoCheck && uniqueVideoCheck.checked) {
        dedupMode = "unique";
    } else if (smartDedupCheck && smartDedupCheck.checked) {
        const selectedRadio = document.querySelector('input[name="smart-submode"]:checked');
        dedupMode = selectedRadio ? selectedRadio.value : "smart";
    }

    if (!qVisual && !qOcr && !qSpeech && !vId && !isObj) {
        if (currentImageQueryBase64) {
            executeImageSearch();
            return;
        }
        statusText.innerText = "⚠️ Vui lòng nhập Visual Query, OCR, Lời thoại hoặc bật Lọc Vật Thể.";
        showToast("Vui lòng nhập Visual Query, OCR, Lời thoại hoặc bật Lọc Vật Thể.", "warning");
        return;
    }

    const imageCount = currentImageQueries.length;
    const hasImageQuery = imageCount > 0;
    const isComposed = hasImageQuery && (!!qVisual || !!qOcr);

    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const rrfK = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;

    searchBtn.disabled = true;
    searchBtn.innerText = "SEARCHING...";
    if (btnSearchImageQuery) {
        btnSearchImageQuery.disabled = true;
        btnSearchImageQuery.innerText = "⏳ ĐANG TÌM...";
    }

    if (isComposed) {
        statusText.innerText = `🖼️ (${imageCount} góc ảnh) + 📝 Đang tìm kiếm KẾT HỢP (Ảnh + Văn bản: '${qVisual}') (${isRrf ? `RRF k=${rrfK}` : "Weighted"})...`;
        showToast(`🖼️ (${imageCount} góc ảnh) + 📝 Đang tìm kiếm kết hợp Đa Phương Thức...`, "loading", 0);
    } else if (hasImageQuery) {
        statusText.innerText = `🖼️ Đang tìm kiếm bằng ${imageCount} góc ảnh (Qwen3-VL 2048D Multi-View)...`;
        showToast(`🖼️ Đang tìm kiếm bằng ${imageCount} góc ảnh (Qwen3-VL 2048D Multi-View)...`, "loading", 0);
    } else {
        statusText.innerText = `Đang tìm kiếm đa kênh (${isRrf ? `RRF k=${rrfK}` : "Weighted"} | Visual: ${isVis ? "ON" : "OFF"}, ASR: ${isAsr ? "ON" : "OFF"}, Caption: ${rrfSignalCaption?.checked ? "ON" : "OFF"})...`;
        showToast(`⏳ Đang tìm kiếm (${isRrf ? `RRF k=${rrfK}` : "Weighted"})...`, "loading", 0);
    }
    resultsGrid.innerHTML = "";

    commitPendingTags();

    const taxonomy = getTaxonomyFilterPayload();

    const payload = {
        query: qVisual,
        image_base64: currentImageQueries.length > 0 ? currentImageQueries[0] : null,
        images_base64: currentImageQueries.length > 0 ? currentImageQueries : null,
        image_weight: 0.5,
        ocr_query: qOcr,
        speech_query: qSpeech,
        enable_asr: isAsr,
        enable_object: isObj,
        object_class_id: isObj ? parseInt(objectClassSelect ? objectClassSelect.value : 0) : null,
        object_conf_thresh: isObj ? parseFloat(objectConfSlider ? objectConfSlider.value : 0.3) : 0.0,
        dedup_mode: dedupMode,
        visual_sim_thresh: parseFloat(visualSimNum ? visualSimNum.value : 0.90),
        enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
        cross_video_sim_thresh: parseFloat(crossVideoSimNum ? crossVideoSimNum.value : 0.90) || 0.90,
        video_id: "",
        include_videos: currentIncludeTags.length > 0 ? currentIncludeTags : null,
        exclude_videos: currentExcludeTags.length > 0 ? currentExcludeTags : null,
        category: taxonomy.categories,
        sub_category: taxonomy.subjects,
        weight_visual: isVis ? parseFloat(weightVisualSlider ? weightVisualSlider.value : 1.0) : 0.0,
        weight_ocr: isOcr ? parseFloat(weightOcrSlider ? weightOcrSlider.value : 1.0) : 0.0,
        weight_asr: isAsr ? parseFloat(weightAsrSlider ? weightAsrSlider.value : 1.0) : 0.0,
        weight_object: isObj ? parseFloat(weightObjectSlider ? weightObjectSlider.value : 0.5) : 0.0,
        top_k: parseInt(topKNum ? topKNum.value : 60) || 60,
        use_rrf: isRrf,
        rrf_k: rrfK,
        asr_window_n: selectedAsrWindowN,
        rrf_use_visual: rrfSignalVisual ? rrfSignalVisual.checked : true,
        rrf_use_asr: rrfSignalAsr ? rrfSignalAsr.checked : true,
        rrf_use_shot: rrfSignalShot ? rrfSignalShot.checked : true,
        rrf_use_caption: rrfSignalCaption ? rrfSignalCaption.checked : false,
        return_shots: false
    };

    const t0 = performance.now();

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();
        currentResults = data.results || [];
        currentResults.forEach((r, i) => {
            r.original_rank = i + 1;
        });
        const elapsedMs = ((performance.now() - t0) / 1000).toFixed(2);
        activeResultsSource = currentResults;
        resultCount.innerText = `${currentResults.length} kết quả Top-K (🖼️ Keyframe)`;
        renderResults(currentResults);

        statusText.innerText = `✅ Xong: ${currentResults.length} Top-K (${elapsedMs}s)`;
        showToast(`✅ Đã tìm thấy ${currentResults.length} kết quả Top-K (${elapsedMs}s)!`, "success");

    } catch (err) {
        dismissLoadingToast();
        statusText.innerText = `[ERROR] ${err.message}`;
        showToast(`Lỗi tìm kiếm: ${err.message}`, "error");
        console.error(err);
    } finally {
        searchBtn.disabled = !backendReady;
        searchBtn.innerText = "🚀 SEARCH NOW";
    }
}

// ==========================================================================
// TỰ ĐỘNG TÍNH TOÁN K2 = N x M VÀ ĐỒNG BỘ NHÃN GIAO DIỆN
// ==========================================================================
function updateK2Calculation() {
    const n = parseInt(rerankTopV ? rerankTopV.value : 5) || 1;
    const m = parseInt(rerankKPerVideo ? rerankKPerVideo.value : 40) || 1;
    const k2 = n * m;

    if (labelK2Total) labelK2Total.innerText = `${k2} frames`;
    if (btnTopVLabel) btnTopVLabel.innerText = n;
    if (btnK2Label) btnK2Label.innerText = k2;
}

if (rerankTopV) rerankTopV.addEventListener("input", updateK2Calculation);
if (rerankKPerVideo) rerankKPerVideo.addEventListener("input", updateK2Calculation);
updateK2Calculation();

// ==========================================================================
// HÀM TÁI XẾP HẠNG (TWO-STAGE RERANK TOP N VIDEOS TỪ TOP K1 BAN ĐẦU)
// ==========================================================================
async function performTwoStageRerank() {
    if (isMultiSceneMode && latestParsedScenes && latestParsedScenes.length >= 2) {
        return performMultiSceneSearch();
    }

    const isVis = enableVisualCheck ? enableVisualCheck.checked : true;
    const isOcr = enableOcrCheck ? enableOcrCheck.checked : true;
    const isAsr = enableAsrCheck ? enableAsrCheck.checked : true;
    const isObj = false;

    const qVisual = isVis && searchInput ? searchInput.value.trim() : "";
    const qOcr = isOcr && ocrInput ? ocrInput.value.trim() : "";
    const qSpeech = isAsr && speechInput ? speechInput.value.trim() : "";

    if (!qVisual && !qOcr && !qSpeech && !isObj) {
        statusText.innerText = "⚠️ Vui lòng nhập Visual Query, OCR, Lời thoại hoặc bật Lọc Vật Thể.";
        showToast("⚠️ Vui lòng nhập Visual Query, OCR, Lời thoại hoặc bật Lọc Vật Thể.", "warning");
        return;
    }

    let dedupMode = "none";
    if (uniqueVideoCheck && uniqueVideoCheck.checked) {
        dedupMode = "unique";
    } else if (smartDedupCheck && smartDedupCheck.checked) {
        const selectedRadio = document.querySelector('input[name="smart-submode"]:checked');
        dedupMode = selectedRadio ? selectedRadio.value : "smart";
    }

    const topK1Val = parseInt(topKNum ? topKNum.value : 100) || 100;
    const topNVal = parseInt(rerankTopV ? rerankTopV.value : 5) || 5;
    const mFramesVal = parseInt(rerankKPerVideo ? rerankKPerVideo.value : 40) || 40;
    const expectedK2 = topNVal * mFramesVal;

    const objClass = isObj ? parseInt(objectClassSelect ? objectClassSelect.value : 0) : null;
    const objConf = isObj ? parseFloat(objectConfSlider ? objectConfSlider.value : 0.3) : 0.0;
    const simThresh = parseFloat(visualSimNum ? visualSimNum.value : 0.90) || 0.90;

    const wVisual = parseFloat(weightVisualSlider ? weightVisualSlider.value : 0.5);
    const wOcr = parseFloat(weightOcrSlider ? weightOcrSlider.value : 0.2);
    const wAsr = parseFloat(weightAsrSlider ? weightAsrSlider.value : 0.2);
    const wObj = parseFloat(weightObjectSlider ? weightObjectSlider.value : 0.1);

    commitPendingTags();

    const taxonomy = getTaxonomyFilterPayload();

    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const rrfK = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;

    rerankBtn.disabled = true;
    rerankBtn.innerText = "⏳ ĐANG RERANK...";
    statusText.innerText = `🎯 Đang lấy Top ${topNVal} Video từ Top ${topK1Val} ban đầu -> Quét ${expectedK2} frames...`;
    showToast(`⏳ Đang Rerank Top ${topNVal} Video (${expectedK2} frames)...`, "loading", 0);

    const t0 = performance.now();

    try {
        const response = await fetch(RERANK_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: qVisual,
                image_base64: currentImageQueryBase64 || null,
                reranker_query: qVisual,
                top_v: topNVal,
                k_per_video: mFramesVal,
                ocr_query: qOcr,
                speech_query: qSpeech,
                enable_asr: isAsr,
                enable_object: isObj,
                object_class_id: objClass,
                object_conf_thresh: objConf,
                weight_visual: wVisual,
                weight_ocr: wOcr,
                weight_asr: wAsr,
                weight_object: wObj,
                dedup_mode: dedupMode,
                visual_sim_thresh: simThresh,
                enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
                cross_video_sim_thresh: parseFloat(crossVideoSimNum ? crossVideoSimNum.value : 0.90) || 0.90,
                video_id: "",
                include_videos: currentIncludeTags.length > 0 ? currentIncludeTags : null,
                exclude_videos: currentExcludeTags.length > 0 ? currentExcludeTags : null,
                category: taxonomy.categories,
                sub_category: taxonomy.subjects,
                top_k: topK1Val,
                use_rrf: isRrf,
                rrf_k: rrfK
            })
        });

        if (!response.ok) {
            throw new Error(`Rerank Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();
        currentResults = data.results || [];
        currentResults.forEach((r, i) => {
            r.original_rank = i + 1;
        });
        activeResultsSource = currentResults;
        const elapsedSec = data.elapsed_sec || ((performance.now() - t0) / 1000).toFixed(2);
        const topVList = (data.top_videos || []).join(", ");

        resultCount.innerText = `${currentResults.length} reranked items`;
        statusText.innerText = `🎯 ĐÃ RERANK: Lọc ${data.total_candidates} frames từ Top ${topNVal} Video [${topVList}] (${elapsedSec}s)!`;
        showToast(`🎯 Đã Rerank: ${currentResults.length} frames từ Top ${topNVal} Video (${elapsedSec}s)!`, "success");

        renderResults(currentResults);

        // Cuộn mượt lên đầu danh sách để xem ngay kết quả Rerank Top 1
        const mainContent = document.querySelector(".main-content");
        if (mainContent) mainContent.scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (err) {
        dismissLoadingToast();
        statusText.innerText = `[ERROR RERANK] ${err.message}`;
        showToast(`Lỗi Rerank: ${err.message}`, "error");
        console.error(err);
    } finally {
        rerankBtn.disabled = false;
        updateK2Calculation();
        rerankBtn.innerText = "🎯 RERANK VIDEO";
    }
}

if (searchBtn) {
    searchBtn.addEventListener("click", performSearch);
}

if (trakeSingleSearchBtn) {
    trakeSingleSearchBtn.addEventListener("click", performSearch);
}

if (rerankBtn) {
    rerankBtn.addEventListener("click", performTwoStageRerank);
}

// ==========================================================================
// RENDER KẾT QUẢ TOP-K BẰNG KEYFRAME TĨNH
// ==========================================================================
function renderResults(results) {
    if (!resultsGrid) return;
    resultsGrid.classList.remove("trake-mode-grid");
    resultsGrid.innerHTML = "";

    if (!results || results.length === 0) {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 40px; text-align: center; color: var(--text-dim); font-size: 0.9rem;">Không tìm thấy kết quả nào thỏa mãn bộ lọc.</div>`;
        return;
    }

    results.forEach((res, idx) => {
        const card = document.createElement("div");
        const isExcluded = currentExcludeTags.includes((res.video_id || "").toUpperCase());
        card.className = isExcluded ? "result-card card-excluded" : "result-card";
        card.dataset.videoId = res.video_id || "";
        if (res.shot_id) card.dataset.shotId = res.shot_id;

        const rankNum = res.original_rank !== undefined ? res.original_rank : (idx + 1);
        const scoreFinal = (res.score !== undefined && res.score !== null) ? Number(res.score).toFixed(4) : "0.0000";
        const scoreVis = (res.visual_score !== undefined && res.visual_score !== null) ? Number(res.visual_score).toFixed(4) : "0.0000";
        const scoreAsr = (res.asr_score !== undefined && res.asr_score !== null) ? Number(res.asr_score).toFixed(4) : "0.0000";
        const scoreOcr = (res.ocr_score !== undefined && res.ocr_score !== null) ? Number(res.ocr_score).toFixed(4) : "0.0000";
        const scoreObj = (res.obj_score !== undefined && res.obj_score !== null) ? Number(res.obj_score).toFixed(4) : "0.0000";

        // Xử lý frame_id
        let bestFid = res.best_kf_frame_id || res.anchor_frame_id || res.frame_id || "";
        if (typeof bestFid === "number") bestFid = String(bestFid).padStart(6, "0");
        card.dataset.frameId = bestFid;
        if (typeof dresState !== "undefined" && dresState.submittedKeys && dresState.submittedKeys.has(res.video_id + '_' + bestFid)) {
            card.classList.add("card-dres-submitted");
        }

        // Xử lý mốc thời gian
        let timeStr = "";
        let timeRangeStr = "";
        if (res.shot_start_sec !== undefined && res.shot_end_sec !== undefined && res.shot_end_sec > 0) {
            const dur = res.shot_duration_sec !== undefined ? Number(res.shot_duration_sec).toFixed(1) : (res.shot_end_sec - res.shot_start_sec).toFixed(1);
            const m1 = Math.floor(res.shot_start_sec / 60).toString().padStart(2, "0");
            const s1 = Math.floor(res.shot_start_sec % 60).toString().padStart(2, "0");
            const m2 = Math.floor(res.shot_end_sec / 60).toString().padStart(2, "0");
            const s2 = Math.floor(res.shot_end_sec % 60).toString().padStart(2, "0");
            timeRangeStr = `${m1}:${s1} ➔ ${m2}:${s2} (${dur}s)`;
        } else if (res.timestamp_sec !== undefined && res.timestamp_sec !== null) {
            const totalSec = Math.floor(res.timestamp_sec);
            const m = Math.floor(totalSec / 60).toString().padStart(2, "0");
            const s = (totalSec % 60).toString().padStart(2, "0");
            timeStr = `${m}:${s}`;
        }

        // Lời thoại ASR Speech Box
        let speechHtml = "";
        if (res.asr_transcript && res.asr_transcript.trim()) {
            const startStr = res.asr_start !== undefined ? Number(res.asr_start).toFixed(1) : "0.0";
            const endStr = res.asr_end !== undefined ? Number(res.asr_end).toFixed(1) : "0.0";
            speechHtml = `
                <div class="speech-box">
                    🎙️ "${res.asr_transcript}"
                    <span class="speech-time">⏱️ [${startStr}s ➔ ${endStr}s]</span>
                </div>
            `;
        }

        const boxCount = (res.boxes && res.boxes.length > 0) ? res.boxes.length : 0;
        const boxBadge = boxCount > 0 ? `<span class="badge-obj">📦 ${boxCount} objs</span>` : "";
        const rerankBadge = res.is_reranked ? `<span class="badge-reranked">🎯 Reranked</span>` : "";
        const sceneBadge = res.scene_name ? `<span class="scene-badge-tag">🎬 ${res.scene_name}</span>` : "";

        const imgUrl = res.image_url || res.thumbnail_url || `/images/${res.video_id}/${bestFid}.jpg`;

        const mediaHtml = `
            <div class="card-thumb-wrapper" style="position: relative; overflow: hidden; cursor: pointer; height: 160px; background: #000;" title="💡 Ctrl + Click: Nộp bài ngay | Click: Phóng to soi chi tiết">
                <img src="${imgUrl}" loading="lazy" alt="Keyframe ${bestFid}" style="width: 100%; height: 100%; object-fit: cover; display: block;">
                <span class="card-score-pill">Score: ${scoreFinal}</span>
                ${timeStr ? `<span class="card-time-pill">⏱️ ${timeStr}</span>` : (timeRangeStr ? `<span class="card-time-pill">⏱️ ${timeRangeStr}</span>` : '')}
                <span class="card-ctrl-hint" title="Giữ phím Ctrl + Click vào ảnh để Nộp bài ngay">⚡ Ctrl+Click: Nộp</span>
                ${dresState.submittedKeys.has(res.video_id + '_' + bestFid) ? `<span class="card-submitted-badge">✓ ĐÃ NỘP DRES</span>` : ''}
            </div>
        `;

        // Minh bạch thứ hạng từng kênh thành phần (Transparency Ranks Breakdown)
        const rrfModeLabel = res.rrf_enabled === false ? "RRF OFF" : `RRF ${scoreFinal}`;
        const rrfModeTitle = res.rrf_enabled === false
            ? "RRF không được dùng; đây là điểm weighted fusion"
            : `Điểm RRF đã chuẩn hóa · raw ${Number(res.raw_rrf || 0).toFixed(4)} · ${res.rrf_channel_count || 0} kênh có dữ liệu`;

        card.innerHTML = `
            ${mediaHtml}
            <div class="card-meta-bar">
                <div class="card-video-row" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; cursor: pointer;" title="Bấm để xem chi tiết trong Lightbox">
                    <span class="card-vid-name" style="color: #38bdf8; font-weight: 700;">#${rankNum} ${res.video_id}</span>
                    <span class="card-frame-id" style="background: rgba(30, 41, 59, 0.8); color: #cbd5e1; font-size: 0.68rem; padding: 2px 6px; border-radius: 4px;">F: ${bestFid}</span>
                </div>

                <!-- Dải minh bạch thứ hạng (Transparency Ranks Strip) -->
                <div class="transparency-ranks-strip" style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 5px; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;">
                    ${renderRrfChannelBadge(res, "visual")}
                    ${renderRrfChannelBadge(res, "asr")}
                    ${renderRrfChannelBadge(res, "shot")}
                    ${renderRrfChannelBadge(res, "caption")}
                    <span class="rank-badge badge-rrf" title="${rrfModeTitle}">⚡ ${rrfModeLabel}</span>
                    ${boxBadge}
                    ${rerankBadge}
                    ${sceneBadge}
                </div>

                ${timeRangeStr ? `<div style="font-size: 0.72rem; color: #94a3b8; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace;">⏱️ ${timeRangeStr}</div>` : ''}
                ${speechHtml}

                <div class="card-actions-row" style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; margin-top: 6px;">
                    <button class="btn-card-action btn-card-zoom" style="background: #0ea5e9; color: white; padding: 6px 2px; font-size: 0.72rem; font-weight: 600;" title="Phóng to xem chi tiết trong Lightbox">🔍 Soi</button>
                    <button class="btn-card-action btn-card-timeline" style="background: #334155; color: white; padding: 6px 2px; font-size: 0.72rem; font-weight: 600;" title="Xem toàn bộ Timeline video">🎞️ Timeline</button>
                    <button class="btn-card-action btn-card-copy" style="background: #1e293b; color: #cbd5e1; padding: 6px 2px; font-size: 0.72rem; font-weight: 600;" title="Copy Video ID & Frame ID">📋 Copy ID</button>
                </div>
                <button class="btn-card-exclude ${isExcluded ? 'is-excluded' : ''}" style="width: 100%; margin-top: 3px;" title="Loại trừ / hoàn tác loại trừ video này khỏi kết quả">
                    ${isExcluded ? '🔴 Đã loại • Bỏ loại trừ' : '🚫 Loại video này'}
                </button>
            </div>
        `;

        // Gắn sự kiện click / Ctrl+Click
        const handleCtrlSubmit = (e) => {
            if (e.ctrlKey || e.metaKey) {
                e.stopPropagation();
                e.preventDefault();
                handleDresFastSubmit(res.video_id, bestFid, card);
                return true;
            }
            return false;
        };

        const mediaBox = card.querySelector(".card-thumb-wrapper");
        const vidRow = card.querySelector(".card-video-row");

        if (mediaBox) {
            mediaBox.addEventListener("click", (e) => {
                if (handleCtrlSubmit(e)) return;
                openLightbox(idx, results);
            });
        }

        if (vidRow) {
            vidRow.addEventListener("click", (e) => {
                if (handleCtrlSubmit(e)) return;
                openLightbox(idx, results);
            });
        }

        card.addEventListener("click", (e) => {
            if (e.ctrlKey || e.metaKey) {
                handleCtrlSubmit(e);
            }
        });

        const btnZoom = card.querySelector(".btn-card-zoom");
        if (btnZoom) {
            btnZoom.addEventListener("click", (e) => {
                e.stopPropagation();
                openLightbox(idx, results);
            });
        }

        const btnTimeline = card.querySelector(".btn-card-timeline");
        if (btnTimeline) {
            btnTimeline.addEventListener("click", (e) => {
                e.stopPropagation();
                openTimeline(res.video_id, bestFid);
            });
        }

        const btnCopy = card.querySelector(".btn-card-copy");
        if (btnCopy) {
            btnCopy.addEventListener("click", (e) => {
                e.stopPropagation();
                const submissionText = `${res.video_id} ${bestFid}`;
                navigator.clipboard.writeText(submissionText).then(() => {
                    btnCopy.innerText = "✅ ĐÃ COPY!";
                    showToast(`📋 Đã sao chép: ${submissionText}`, "success", 1800);
                    setTimeout(() => {
                        btnCopy.innerText = "📋 Copy ID";
                    }, 1500);
                });
            });
        }

        const excludeBtn = card.querySelector(".btn-card-exclude");
        if (excludeBtn) {
            excludeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                const vidUpper = (res.video_id || "").toUpperCase();
                const eIdx = currentExcludeTags.indexOf(vidUpper);
                if (eIdx === -1) {
                    currentExcludeTags.push(vidUpper);
                    showToast(`🚫 Đã loại trừ video ${res.video_id} khỏi danh sách`, "warning", 2000);
                } else {
                    currentExcludeTags.splice(eIdx, 1);
                    showToast(`✅ Đã bỏ loại trừ video ${res.video_id}`, "success", 2000);
                }
                renderExcludeTags();
                applyExcludeStateToGrid();
            });
        }

        resultsGrid.appendChild(card);
    });
}

// ==========================================================================
// APPLY EXCLUDE STATE TO GRID (không re-render, chỉ cập nhật CSS trực tiếp)
// ==========================================================================
function applyExcludeStateToGrid() {
    if (!resultsGrid) return;

    // 1. Cập nhật các thẻ kết quả chế độ KIS / QA
    const cards = resultsGrid.querySelectorAll(".result-card");
    cards.forEach(card => {
        const videoId = (card.dataset.videoId || "").toUpperCase();
        const isExcluded = currentExcludeTags.includes(videoId);
        if (isExcluded) {
            card.classList.add("card-excluded");
        } else {
            card.classList.remove("card-excluded");
        }
        const excludeBtn = card.querySelector(".btn-card-exclude");
        if (excludeBtn) {
            if (isExcluded) {
                excludeBtn.classList.add("is-excluded");
                excludeBtn.textContent = "🔴 Đã loại • Bỏ loại trừ";
            } else {
                excludeBtn.classList.remove("is-excluded");
                excludeBtn.textContent = "🚫 Loại video này";
            }
        }
    });

    // 2. Cập nhật các thẻ chuỗi phân cảnh chế độ TRAKE Storyboard
    const trakeCards = resultsGrid.querySelectorAll(".trake-group-card");
    trakeCards.forEach(card => {
        const videoId = (card.dataset.videoId || "").toUpperCase();
        const isExcluded = currentExcludeTags.includes(videoId);
        if (isExcluded) {
            card.classList.add("card-excluded");
        } else {
            card.classList.remove("card-excluded");
        }
        const excludeBtn = card.querySelector(".btn-sb-exclude");
        if (excludeBtn) {
            if (isExcluded) {
                excludeBtn.classList.add("is-excluded");
                excludeBtn.textContent = "🔴 Đã loại • Bỏ loại";
            } else {
                excludeBtn.classList.remove("is-excluded");
                excludeBtn.textContent = "🚫 Loại video";
            }
        }
    });
}
window.applyExcludeStateToGrid = applyExcludeStateToGrid;

function showLightboxKeyframe(item, bestFid) {
    if (!item) return;
    if (lightboxZoomBox) lightboxZoomBox.style.display = "block";
    if (lightboxCanvasControls) lightboxCanvasControls.style.display = "flex";

    const imgUrl = item.thumbnail_url || item.image_url || `/images/${item.video_id}/${bestFid}.jpg`;
    if (lightboxImg) {
        lightboxImg.src = imgUrl;
        if (lightboxImg.complete) {
            setTimeout(drawLightboxCanvas, 50);
        } else {
            lightboxImg.onload = () => setTimeout(drawLightboxCanvas, 50);
        }
    }
}

async function openLightbox(index, sourceList = currentResults) {
    activeResultsSource = sourceList;
    if (!activeResultsSource || index < 0 || index >= activeResultsSource.length) return;

    resetZoom();

    currentLightboxIndex = index;
    const item = activeResultsSource[currentLightboxIndex];
    let bestFid = item.best_kf_frame_id || item.anchor_frame_id || item.frame_id || "";
    if (typeof bestFid === "number") {
        bestFid = String(bestFid).padStart(6, "0");
    }

    const rankNum = item.original_rank !== undefined ? item.original_rank : (index + 1);
    const shotIdxStr = item.shot_index ? `#${item.shot_index}` : (item.shot_id ? item.shot_id : "");
    lightboxTitle.innerText = `${item.video_id} / F: ${bestFid} ${shotIdxStr ? '(' + shotIdxStr + ')' : ''}`;

    const scoreVal = item.score !== undefined ? Number(item.score).toFixed(4) : "1.0000";
    const fusionLabel = (item.rrf_enabled ?? Boolean(useRrfCheck?.checked)) ? "RRF" : "Weighted";
    lightboxScoreBadge.innerText = `Rank #${rankNum} • ${fusionLabel}: ${scoreVal}`;

    const sSec = item.shot_start_sec !== undefined ? item.shot_start_sec : (item.start_sec !== undefined ? item.start_sec : null);
    const eSec = item.shot_end_sec !== undefined ? item.shot_end_sec : (item.end_sec !== undefined ? item.end_sec : null);
    if (sSec !== null && eSec !== null && eSec > 0) {
        const dur = (item.shot_duration_sec || item.duration_sec || (eSec - sSec)).toFixed(1);
        const m1 = Math.floor(sSec / 60).toString().padStart(2, "0");
        const s1 = Math.floor(sSec % 60).toString().padStart(2, "0");
        const m2 = Math.floor(eSec / 60).toString().padStart(2, "0");
        const s2 = Math.floor(eSec % 60).toString().padStart(2, "0");
        lightboxTimeBadge.innerText = `⏱️ ${m1}:${s1} ➔ ${m2}:${s2} (${dur}s)`;
    } else {
        const sec = item.timestamp_sec || 0;
        const mins = Math.floor(sec / 60);
        const secs = Math.floor(sec % 60);
        lightboxTimeBadge.innerText = `⏱️ ${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')} (${sec}s)`;
    }

    // Hiển thị dải huy hiệu minh bạch thứ hạng thành phần trong Lightbox Sidebar
    const ranksStrip = document.getElementById("lightbox-ranks-strip");
    if (ranksStrip) {
        ranksStrip.innerHTML = `
            ${renderRrfChannelBadge(item, "visual", true)}
            ${renderRrfChannelBadge(item, "asr", true)}
            ${renderRrfChannelBadge(item, "shot", true)}
            ${renderRrfChannelBadge(item, "caption", true)}
        `;
    }

    if (copyStatus) copyStatus.innerText = "";

    let asrDisplay = "Không có lời thoại phân đoạn này.";
    if (item.asr_transcript) {
        asrDisplay = `"${item.asr_transcript}"`;
        if (item.asr_prev || item.asr_next) {
            asrDisplay += `\n\n🎙️ [Ngữ cảnh mở rộng ±${selectedAsrWindowN} segments]:\n`;
            if (item.asr_prev) asrDisplay += `⏪ Trước: "${item.asr_prev}"\n`;
            if (item.asr_next) asrDisplay += `⏩ Sau: "${item.asr_next}"`;
        }
    }
    lightboxTranscript.innerText = asrDisplay;

    // Helper render Box Pills
    function renderBoxPills(boxes) {
        if (!lightboxBoxesList) return;
        lightboxBoxesList.innerHTML = "";
        if (boxes && boxes.length > 0) {
            boxes.forEach(b => {
                const pill = document.createElement("span");
                pill.className = "box-pill";
                pill.innerText = `${b.class_name}: ${(b.confidence * 100).toFixed(0)}%`;
                lightboxBoxesList.appendChild(pill);
            });
        } else {
            lightboxBoxesList.innerHTML = `<span style="font-size: 0.8rem; color: #888;">Không phát hiện vật thể nổi bật.</span>`;
        }
    }

    renderBoxPills([]);

    // Mở modal trước để layout tính đúng clientWidth / clientHeight
    lightboxModal.style.display = "flex";

    showLightboxKeyframe(item, bestFid);

}

function closeLightbox() {
    resetZoom();
    if (lightboxModal) lightboxModal.style.display = "none";
}

function drawLightboxCanvas() {
    if (!lightboxImg || !lightboxCanvas) return;

    const item = activeResultsSource[currentLightboxIndex];
    if (!item) return;

    const displayW = lightboxImg.clientWidth;
    const displayH = lightboxImg.clientHeight;
    const naturalW = lightboxImg.naturalWidth || 1280;
    const naturalH = lightboxImg.naturalHeight || 720;

    lightboxCanvas.width = displayW;
    lightboxCanvas.height = displayH;

    const ctx = lightboxCanvas.getContext("2d");
    ctx.clearRect(0, 0, displayW, displayH);

    if (!lightboxToggleBoxes.checked || !item.boxes || item.boxes.length === 0) return;

    const scaleX = displayW / naturalW;
    const scaleY = displayH / naturalH;

    item.boxes.forEach((b) => {
        const [x1, y1, x2, y2] = b.box;
        const drawX = x1 * scaleX;
        const drawY = y1 * scaleY;
        const drawW = (x2 - x1) * scaleX;
        const drawH = (y2 - y1) * scaleY;

        ctx.strokeStyle = "#ff6d00";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(drawX, drawY, drawW, drawH);

        const label = `${b.class_name} ${(b.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 11px 'Space Mono', monospace";
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = "#ff6d00";
        ctx.fillRect(drawX, Math.max(0, drawY - 18), textWidth + 8, 18);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, drawX + 4, Math.max(12, drawY - 4));
    });
}

// ==========================================================================
// INTERACTIVE PAN & ZOOM (MOUSE WHEEL ZOOM AT CURSOR & DRAG TO PAN)
// ==========================================================================
let zoomScale = 1.0;
let zoomTranslateX = 0;
let zoomTranslateY = 0;
let isZoomDragging = false;
let zoomStartX = 0;
let zoomStartY = 0;

const lightboxMediaContainer = document.getElementById("lightbox-media-container");
const lightboxZoomBox = document.getElementById("lightbox-zoom-box");
const lightboxZoomBadge = document.getElementById("lightbox-zoom-badge");
const lightboxZoomResetBtn = document.getElementById("lightbox-zoom-reset-btn");

function updateZoomTransform() {
    if (!lightboxZoomBox) return;
    lightboxZoomBox.style.transform = `translate(${zoomTranslateX}px, ${zoomTranslateY}px) scale(${zoomScale})`;
    if (lightboxZoomBadge) {
        lightboxZoomBadge.innerText = `🔍 ${(zoomScale * 100).toFixed(0)}% (Lăn chuột để Zoom / Kéo để Pan)`;
    }
}

function resetZoom() {
    zoomScale = 1.0;
    zoomTranslateX = 0;
    zoomTranslateY = 0;
    isZoomDragging = false;
    if (lightboxMediaContainer) lightboxMediaContainer.classList.remove("is-dragging");
    updateZoomTransform();
}

if (lightboxMediaContainer) {
    // Wheel Zoom toward Cursor
    lightboxMediaContainer.addEventListener("wheel", (e) => {
        e.preventDefault();
        const rect = lightboxMediaContainer.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomDelta = e.deltaY < 0 ? 1.18 : 0.85;
        const nextScale = Math.min(Math.max(1.0, zoomScale * zoomDelta), 10.0);

        if (nextScale === 1.0) {
            resetZoom();
            return;
        }

        // Adjust translate so the point under the cursor remains fixed
        zoomTranslateX = mouseX - (mouseX - zoomTranslateX) * (nextScale / zoomScale);
        zoomTranslateY = mouseY - (mouseY - zoomTranslateY) * (nextScale / zoomScale);
        zoomScale = nextScale;

        updateZoomTransform();
    }, { passive: false });

    // Drag to Pan
    lightboxMediaContainer.addEventListener("mousedown", (e) => {
        if (zoomScale <= 1.0) return;
        isZoomDragging = true;
        zoomStartX = e.clientX - zoomTranslateX;
        zoomStartY = e.clientY - zoomTranslateY;
        lightboxMediaContainer.classList.add("is-dragging");
    });

    window.addEventListener("mousemove", (e) => {
        if (!isZoomDragging) return;
        zoomTranslateX = e.clientX - zoomStartX;
        zoomTranslateY = e.clientY - zoomStartY;
        updateZoomTransform();
    });

    window.addEventListener("mouseup", () => {
        if (isZoomDragging) {
            isZoomDragging = false;
            if (lightboxMediaContainer) lightboxMediaContainer.classList.remove("is-dragging");
        }
    });

    // Double click to toggle Zoom / Reset
    lightboxMediaContainer.addEventListener("dblclick", (e) => {
        if (zoomScale > 1.0) {
            resetZoom();
        } else {
            const rect = lightboxMediaContainer.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            zoomScale = 2.5;
            zoomTranslateX = mouseX - (mouseX * 2.5);
            zoomTranslateY = mouseY - (mouseY * 2.5);
            updateZoomTransform();
        }
    });
}

if (lightboxZoomResetBtn) {
    lightboxZoomResetBtn.addEventListener("click", resetZoom);
}

// Video Button in Lightbox
if (lightboxVideoBtn) {
    lightboxVideoBtn.addEventListener("click", () => {
        const item = activeResultsSource[currentLightboxIndex];
        if (item && item.video_id) {
            const fid = item.best_kf_frame_id || item.anchor_frame_id || item.frame_id;
            const tSec = (item.start_sec !== undefined) ? item.start_sec : (item.timestamp_sec || 0);
            openVideoPlayer(item.video_id, fid, tSec);
        }
    });
}

// Copy ID Button Handler
if (lightboxCopyBtn) {
    lightboxCopyBtn.addEventListener("click", () => {
        const item = activeResultsSource[currentLightboxIndex];
        if (item) {
            const fid = item.best_kf_frame_id || item.anchor_frame_id || item.frame_id;
            const textToCopy = `${item.video_id} ${fid}`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                if (copyStatus) {
                    copyStatus.innerText = `✅ Đã chép: ${textToCopy}`;
                    setTimeout(() => { if (copyStatus) copyStatus.innerText = ""; }, 2500);
                }
            }).catch(err => {
                if (copyStatus) copyStatus.innerText = `Lỗi copy: ${err}`;
            });
        }
    });
}

// Timeline Button in Lightbox
if (lightboxTimelineBtn) {
    lightboxTimelineBtn.addEventListener("click", () => {
        const item = activeResultsSource[currentLightboxIndex];
        if (item && item.video_id) {
            const fid = item.best_kf_frame_id || item.anchor_frame_id || item.frame_id;
            closeLightbox(); // Tự động đóng/ẩn Lightbox để Timeline hiển thị ngay lập tức lên phía trước!
            openTimeline(item.video_id, fid);
        }
    });
}

// ==========================================================================
// VIDEO PLAYER (TRÌNH PHÁT VIDEO HTML5 TỰ ĐỘNG SEEK ĐẾN KEYFRAME)
// ==========================================================================
function _cancelVideoBadgeLoop() {
    if (_videoBadgeRafId !== null) {
        cancelAnimationFrame(_videoBadgeRafId);
        _videoBadgeRafId = null;
    }
}

function _updateLiveVideoBadge(force = false) {
    if (!html5VideoPlayer || !currentPlayingVideoId) return;
    const now = performance.now();
    if (force || now - _lastBadgeUpdate >= 200) {
        _lastBadgeUpdate = now;
        const fps = _videoFpsCache[String(currentPlayingVideoId).toUpperCase()] || 25.0;
        const currentSec = Number(html5VideoPlayer.currentTime) || 0;
        const currentMs = Math.round(currentSec * 1000);
        const currentFrame = Math.round(currentSec * fps);
        const mins = String(Math.floor(currentSec / 60)).padStart(2, "0");
        const secs = String(Math.floor(currentSec % 60)).padStart(2, "0");
        if (videoTimeBadge) {
            videoTimeBadge.innerText = `⏱️ ${mins}:${secs} (${currentMs}ms) F:${String(currentFrame).padStart(6, "0")}`;
        }
    }
    if (!html5VideoPlayer.paused && !html5VideoPlayer.ended) {
        _videoBadgeRafId = requestAnimationFrame(() => _updateLiveVideoBadge(false));
    } else {
        _videoBadgeRafId = null;
    }
}

function _flashVideoOutline(color, durationMs = 1000) {
    if (!html5VideoPlayer) return;
    if (_videoOutlineTimer) clearTimeout(_videoOutlineTimer);
    html5VideoPlayer.style.outline = `3px solid ${color}`;
    _videoOutlineTimer = setTimeout(() => {
        if (html5VideoPlayer) html5VideoPlayer.style.outline = "";
        _videoOutlineTimer = null;
    }, durationMs);
}

function openVideoPlayer(videoId, frameId, timestampSec, leadInSec = 1.5) {
    if (!videoId) return;
    currentPlayingVideoId = videoId;
    currentPlayingFrameId = frameId;
    currentPlayingTimestamp = (timestampSec !== undefined && timestampSec !== null) ? Number(timestampSec) : 0;
    void getVideoFpsWithCache(videoId).then(() => _updateLiveVideoBadge(true));

    const mins = Math.floor(currentPlayingTimestamp / 60);
    const secs = Math.floor(currentPlayingTimestamp % 60);
    const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;

    if (videoPlayerTitle) videoPlayerTitle.innerText = `▶️ VIDEO: ${videoId}`;
    if (videoTimeBadge) videoTimeBadge.innerText = `⏱️ Keyframe: ${timeStr} (${currentPlayingTimestamp}s)`;

    if (html5VideoPlayer) {
        html5VideoPlayer.onerror = () => {
            if (currentPlayingVideoId !== videoId) return;
            showToast(`❌ Không tìm thấy hoặc không phát được video: ${videoId}`, "error", 4000);
            closeVideoPlayer();
        };
        html5VideoPlayer.onloadedmetadata = () => {
            // Luồng kết quả giữ lead-in 1.5s; Quick Opener/Event Jump truyền 0 để đến đúng mốc.
            html5VideoPlayer.currentTime = Math.max(0, currentPlayingTimestamp - Math.max(0, Number(leadInSec) || 0));
            html5VideoPlayer.playbackRate = parseFloat(videoSpeedSelect ? videoSpeedSelect.value : 1.0);
            html5VideoPlayer.play().catch(err => console.log("Autoplay blocked:", err));
        };
        html5VideoPlayer.src = `/api/video-file/${encodeURIComponent(videoId)}`;
    }

    if (videoModal) videoModal.style.display = "flex";
}

function closeVideoPlayer() {
    _cancelVideoBadgeLoop();
    if (_videoOutlineTimer) {
        clearTimeout(_videoOutlineTimer);
        _videoOutlineTimer = null;
    }
    if (html5VideoPlayer) {
        html5VideoPlayer.pause();
        html5VideoPlayer.onerror = null;
        html5VideoPlayer.onloadedmetadata = null;
        html5VideoPlayer.style.outline = "";
        html5VideoPlayer.removeAttribute("src");
        html5VideoPlayer.load();
    }
    if (videoModal) videoModal.style.display = "none";
}

if (videoCloseBtn) videoCloseBtn.addEventListener("click", closeVideoPlayer);

if (videoModal) {
    videoModal.addEventListener("click", (e) => {
        if (e.target === videoModal) closeVideoPlayer();
    });
}

const videoStartBtn = document.getElementById("video-start-btn");
if (videoStartBtn) {
    videoStartBtn.addEventListener("click", () => {
        if (html5VideoPlayer) html5VideoPlayer.currentTime = 0;
    });
}

if (videoRewind5s) {
    videoRewind5s.addEventListener("click", () => {
        if (html5VideoPlayer) html5VideoPlayer.currentTime = Math.max(0, html5VideoPlayer.currentTime - 5);
    });
}

if (videoForward5s) {
    videoForward5s.addEventListener("click", () => {
        if (html5VideoPlayer) html5VideoPlayer.currentTime = html5VideoPlayer.currentTime + 5;
    });
}

if (videoJumpKeyframe) {
    videoJumpKeyframe.addEventListener("click", () => {
        if (html5VideoPlayer) html5VideoPlayer.currentTime = currentPlayingTimestamp;
    });
}

if (videoSpeedSelect) {
    videoSpeedSelect.addEventListener("change", () => {
        if (html5VideoPlayer) html5VideoPlayer.playbackRate = parseFloat(videoSpeedSelect.value);
    });
}

if (html5VideoPlayer) {
    html5VideoPlayer.addEventListener("play", () => {
        _cancelVideoBadgeLoop();
        _videoBadgeRafId = requestAnimationFrame(() => _updateLiveVideoBadge(true));
    });
    html5VideoPlayer.addEventListener("pause", () => {
        _cancelVideoBadgeLoop();
        _updateLiveVideoBadge(true);
    });
    html5VideoPlayer.addEventListener("ended", _cancelVideoBadgeLoop);
    html5VideoPlayer.addEventListener("seeked", () => _updateLiveVideoBadge(true));

    html5VideoPlayer.addEventListener("click", async (event) => {
        if (!(event.ctrlKey || event.metaKey) || !currentPlayingVideoId) return;
        event.preventDefault();
        event.stopPropagation();

        const currentSec = Number(html5VideoPlayer.currentTime) || 0;
        const fps = await getVideoFpsWithCache(currentPlayingVideoId);
        // Video cho phep chon frame chinh xac nam GIUA hai keyframe da trich xuat.
        // Khong snap ve timeline KF, vi do thua sampling chinh la nguon sai lech 0.5-1s.
        const frameId = String(Math.round(currentSec * fps)).padStart(6, "0");
        const currentMode = dresTaskModeSelect ? dresTaskModeSelect.value : "kis";

        if (currentMode === "kis") {
            html5VideoPlayer.pause();
            _flashVideoOutline("#10b981", 1200);
            showToast(`⚡ KIS: ${currentPlayingVideoId} / F:${frameId} (${currentSec.toFixed(2)}s)`, "info", 1500);
            await triggerDresSubmit(currentPlayingVideoId, frameId, null);
            return;
        }

        if (currentMode === "qa") {
            html5VideoPlayer.pause();
            _flashVideoOutline("#38bdf8", 1200);
            handleDresFastSubmit(currentPlayingVideoId, frameId, null);
            return;
        }

        if (currentMode === "trake") {
            if (trakeTimelineState.videoId
                && trakeTimelineState.videoId !== currentPlayingVideoId
                && trakeTimelineState.selectedFrames.length > 0) {
                const shouldReplace = confirm(
                    `Chuỗi TRAKE hiện tại thuộc "${trakeTimelineState.videoId}". Chuyển sang "${currentPlayingVideoId}" và xóa chuỗi cũ?`
                );
                if (!shouldReplace) return;
                trakeTimelineState.selectedFrames = [];
            }

            trakeTimelineState.videoId = currentPlayingVideoId;
            const existingIdx = trakeTimelineState.selectedFrames.findIndex(
                item => String(item.frame_id) === frameId
            );
            if (existingIdx >= 0) {
                trakeTimelineState.selectedFrames.splice(existingIdx, 1);
                _flashVideoOutline("#ef4444", 800);
                showToast(`↺ Bỏ chọn F:${frameId} khỏi chuỗi TRAKE`, "info", 1200);
            } else {
                trakeTimelineState.selectedFrames.push({
                    frame_id: frameId,
                    timestamp_sec: Number(currentSec.toFixed(3)),
                    element: null
                });
                _flashVideoOutline("#a855f7", 800);
                showToast(
                    `✨ E${trakeTimelineState.selectedFrames.length}: F:${frameId} (${currentSec.toFixed(2)}s) đã thêm`,
                    "success",
                    1500
                );
            }
            if (trakeTimelineBar) trakeTimelineBar.style.display = "flex";
            updateTrakeTimelineUi();
        }
    });
}

if (videoCopyBtn) {
    videoCopyBtn.addEventListener("click", () => {
        if (currentPlayingVideoId && currentPlayingFrameId) {
            const textToCopy = `${currentPlayingVideoId} ${currentPlayingFrameId}`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                showToast(`📋 Đã chép: ${textToCopy}`, "success", 1500);
            });
        }
    });
}

let activeQuickVideoTab = "verify";

function setQuickVideoTab(tabName) {
    activeQuickVideoTab = tabName === "submit" ? "submit" : "verify";
    quickVideoTabs.forEach(tab => tab.classList.toggle("active", tab.dataset.quickTab === activeQuickVideoTab));
    quickVideoVerifyPanel?.classList.toggle("active", activeQuickVideoTab === "verify");
    quickVideoSubmitPanel?.classList.toggle("active", activeQuickVideoTab === "submit");
}

function setQuickVideoMapStatus(message, kind = "") {
    if (!quickVideoMapStatus) return;
    quickVideoMapStatus.innerText = message;
    quickVideoMapStatus.classList.toggle("is-ok", kind === "ok");
    quickVideoMapStatus.classList.toggle("is-warning", kind === "warning");
}

function openQuickVideoOpener(initialTab = "verify") {
    if (!quickVideoOverlay) return;
    setQuickVideoTab(initialTab);
    quickVideoOverlay.style.display = "flex";
    if (currentPlayingVideoId && quickVideoIdInput && !quickVideoIdInput.value.trim()) {
        quickVideoIdInput.value = currentPlayingVideoId;
    }
    setTimeout(() => {
        if (quickVideoIdInput?.value.trim()) quickVideoTimeInput?.focus();
        else quickVideoIdInput?.focus();
    }, 60);
}

function closeQuickVideoOpener() {
    if (quickVideoOverlay) quickVideoOverlay.style.display = "none";
}

function normalizeQuickVideoId(rawVideoId) {
    const videoId = String(rawVideoId || "").trim().toUpperCase().replace(/\.MP4$/i, "");
    return /^[A-Z0-9_-]+$/.test(videoId) ? videoId : "";
}

async function parseQuickFrameInput(rawValue, videoId) {
    const value = String(rawValue || "").trim().toLowerCase();
    if (!value) return [];

    // Danh sach so cach nhau bang dau phay duoc hieu la chuoi KF TRAKE.
    if (value.includes(",")) {
        const ids = value.split(",").map(item => item.trim().replace(/^f[:\s]?/, ""));
        if (ids.some(item => !/^\d+$/.test(item))) return null;
        return ids.map(item => Number.parseInt(item, 10));
    }

    const frameMatch = value.match(/^f[:\s]?(\d+)$/) || value.match(/^(\d+)$/);
    if (frameMatch) return [Number.parseInt(frameMatch[1], 10)];

    let timeSec = null;
    const explicitSeconds = value.match(/^t[:\s]?(\d+(?:\.\d+)?)$/);
    const mmssMatch = value.match(/^(\d+):(\d{1,2}(?:\.\d+)?)$/);
    if (explicitSeconds) {
        timeSec = Number.parseFloat(explicitSeconds[1]);
    } else if (mmssMatch) {
        const seconds = Number.parseFloat(mmssMatch[2]);
        if (seconds >= 60) return null;
        timeSec = Number.parseInt(mmssMatch[1], 10) * 60 + seconds;
    }
    if (!Number.isFinite(timeSec) || timeSec < 0) return null;
    const fps = await getVideoFpsWithCache(videoId);
    return [Math.round(timeSec * fps)];
}

async function resolveFrameAgainstMap(videoId, frameId = null) {
    const suffix = frameId === null ? "" : `?frame_id=${encodeURIComponent(frameId)}`;
    const response = await fetch(`${KEYFRAME_RESOLVE_API}/${encodeURIComponent(videoId)}${suffix}`);
    if (!response.ok) throw new Error(`Resolver HTTP ${response.status}`);
    const data = await response.json();
    if (data.status !== "success") throw new Error(data.message || "Không đối chiếu được keyframe map.");
    return data;
}

function renderResolvedMapStatus(resolutions) {
    if (!resolutions.length) return;
    const first = resolutions[0];
    if (!first.video_file_exists && !first.keyframe_map_exists) {
        setQuickVideoMapStatus(`Không tìm thấy video ${first.video_id} trong video gốc hoặc keyframe map.`, "warning");
        return;
    }
    const fpsText = first.actual_video_fps
        ? `${first.mapped_fps} FPS map / ${first.actual_video_fps} FPS MP4`
        : `${first.mapped_fps} FPS map`;
    const frameText = resolutions.map(item => {
        if (!item.requested_frame_id) return "không chỉ định KF";
        return item.exact_match
            ? `F:${item.resolved_frame_id} ✓`
            : `F:${item.requested_frame_id} → gần nhất F:${item.resolved_frame_id} (Δ${item.frame_delta})`;
    }).join(" · ");
    const warning = first.fps_mismatch || resolutions.some(item => item.requested_frame_id && !item.exact_match);
    setQuickVideoMapStatus(`${first.video_id} · ${frameText} · ${fpsText}`, warning ? "warning" : "ok");
}

async function resolveQuickVideoTargets({ allowMultiple = false } = {}) {
    const videoId = normalizeQuickVideoId(quickVideoIdInput?.value);
    if (!videoId) {
        showToast("⚠️ Video ID không hợp lệ. Ví dụ: L26_V001", "warning");
        quickVideoIdInput?.focus();
        return null;
    }
    const frameIds = await parseQuickFrameInput(quickVideoTimeInput?.value, videoId);
    if (frameIds === null || (!allowMultiple && frameIds.length > 1)) {
        showToast("⚠️ KF không hợp lệ. Dùng 003600, F:3600, T:150 hoặc 02:30.", "warning");
        quickVideoTimeInput?.focus();
        return null;
    }

    setQuickVideoMapStatus("Đang đối chiếu keyframe map và FPS MP4...");
    try {
        const requested = frameIds.length ? frameIds : [null];
        const resolutions = await Promise.all(requested.map(frameId => resolveFrameAgainstMap(videoId, frameId)));
        renderResolvedMapStatus(resolutions);
        if (!resolutions[0].video_file_exists && !resolutions[0].keyframe_map_exists) {
            showToast(`⚠️ Không tìm thấy video ${videoId}.`, "warning");
            return null;
        }
        const outOfRange = resolutions.find(item => item.requested_in_video_range === false);
        if (outOfRange) {
            showToast(
                `⚠️ F:${outOfRange.requested_frame_id} nằm ngoài video (${outOfRange.video_frame_count} frames).`,
                "warning"
            );
            return null;
        }
        const first = resolutions[0];
        const submissionFrameIds = resolutions
            .map(item => item.requested_frame_id || item.resolved_frame_id)
            .filter(Boolean);
        const firstFrameNumber = submissionFrameIds.length ? Number.parseInt(submissionFrameIds[0], 10) : null;
        return {
            videoId,
            resolutions,
            // Nop/mo video giu dung frame nguoi dung nhap; nearest KF chi de
            // doi chieu va lam dich cuon timeline.
            frameIds: submissionFrameIds,
            frameId: submissionFrameIds[0] || null,
            timelineFrameId: first.resolved_frame_id || submissionFrameIds[0] || null,
            timeSec: firstFrameNumber === null ? 0 : firstFrameNumber / Number(first.mapped_fps || 25)
        };
    } catch (error) {
        setQuickVideoMapStatus(error.message, "warning");
        showToast(`⚠️ ${error.message}`, "warning");
        return null;
    }
}

window.addEventListener("keydown", (event) => {
    if (event.altKey && event.code === "KeyG") {
        event.preventDefault();
        if (quickVideoOverlay?.style.display === "flex") closeQuickVideoOpener();
        else openQuickVideoOpener();
        return;
    }
    if (event.key === "Escape" && quickVideoOverlay?.style.display === "flex") {
        event.preventDefault();
        closeQuickVideoOpener();
    }
});

quickVideoOpenBtn?.addEventListener("click", () => openQuickVideoOpener("verify"));
quickVideoCloseBtn?.addEventListener("click", closeQuickVideoOpener);
quickVideoTabs.forEach(tab => tab.addEventListener("click", () => setQuickVideoTab(tab.dataset.quickTab)));

quickVideoOverlay?.addEventListener("click", (event) => {
    if (event.target === quickVideoOverlay) closeQuickVideoOpener();
});

quickVideoIdInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        quickVideoTimeInput?.focus();
    }
});

quickVideoTimeInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        if (activeQuickVideoTab === "submit") btnQuickVideoSubmit?.click();
        else btnQuickVideoGo?.click();
    }
});

btnQuickVideoGo?.addEventListener("click", async () => {
    const target = await resolveQuickVideoTargets();
    if (!target) return;
    closeQuickVideoOpener();
    openVideoPlayer(target.videoId, target.frameId, target.timeSec, 0);
});

btnQuickVideoTimeline?.addEventListener("click", async () => {
    const target = await resolveQuickVideoTargets();
    if (!target) return;
    closeQuickVideoOpener();
    if (videoModal?.style.display !== "none") closeVideoPlayer();
    if (lightboxModal?.style.display !== "none") closeLightbox();
    openTimeline(target.videoId, target.timelineFrameId);
});

btnQuickVideoSubmit?.addEventListener("click", async () => {
    const mode = quickSubmitMode?.value || "kis";
    const target = await resolveQuickVideoTargets({ allowMultiple: mode === "trake" });
    if (!target) return;

    if (mode === "trake") {
        if (target.frameIds.length < 2) {
            showToast("⚠️ TRAKE cần ít nhất 2 KF ID cách nhau bằng dấu phẩy.", "warning");
            return;
        }
        closeQuickVideoOpener();
        triggerDresTrakeSubmit(target.videoId, target.frameIds, null);
        return;
    }
    if (!target.frameId) {
        showToast("⚠️ Hãy nhập KF ID trước khi nộp.", "warning");
        return;
    }

    closeQuickVideoOpener();
    if (mode === "qa") {
        openQaSubmissionModal(target.videoId, target.frameId, null);
    } else {
        triggerDresSubmit(target.videoId, target.frameId, null);
    }
});

// ==========================================================================
// TIMELINE BROWSER (CHIẾN THUẬT 5 + AUTO-SCROLL & HIGHLIGHT ĐẾN ĐÚNG FRAME)
// ==========================================================================
// ==========================================================================
// TIMELINE BROWSER (CHIẾN THUẬT 5 + TRAKE EVENT SELECTION + AUTO-SCROLL)
// ==========================================================================
let trakeTimelineState = {
    videoId: "",
    selectedFrames: [] // [{ frame_id, timestamp_sec, element }]
};

const trakeTimelineBar = document.getElementById("trake-timeline-bar");
const trakeSelectedChainText = document.getElementById("trake-selected-chain-text");
const trakeEventPills = document.getElementById("trake-event-pills");
const btnTrakeSubmitDres = document.getElementById("btn-trake-submit-dres");
const btnTrakeClearSelection = document.getElementById("btn-trake-clear-selection");

function updateTrakeTimelineUi() {
    if (!trakeTimelineBar) return;

    // Xóa toàn bộ badge cũ và class trake-selected
    document.querySelectorAll(".timeline-item").forEach(item => {
        item.classList.remove("trake-selected");
        const existingBadge = item.querySelector(".timeline-trake-badge");
        if (existingBadge) existingBadge.remove();
    });

    // Đánh số neon E1, E2, E3... theo thứ tự thời gian đã chọn
    trakeTimelineState.selectedFrames.forEach((item, idx) => {
        const el = item.element || document.querySelector(`.timeline-item[data-frame-id="${item.frame_id}"]`);
        if (el) {
            el.classList.add("trake-selected");
            const thumbBox = el.querySelector(".timeline-thumb-box");
            if (thumbBox && !thumbBox.querySelector(".timeline-trake-badge")) {
                const b = document.createElement("span");
                b.className = "timeline-trake-badge";
                b.innerText = `E${idx + 1}`;
                thumbBox.appendChild(b);
            }
        }
    });

    if (trakeEventPills) {
        trakeEventPills.innerHTML = "";
        trakeTimelineState.selectedFrames.forEach((item, idx) => {
            const timestampSec = Number(item.timestamp_sec) || 0;
            const mins = String(Math.floor(timestampSec / 60)).padStart(2, "0");
            const secs = String(Math.floor(timestampSec % 60)).padStart(2, "0");
            const pill = document.createElement("button");
            pill.type = "button";
            pill.className = "trake-event-pill";
            pill.innerText = `E${idx + 1} ${mins}:${secs} →`;
            pill.title = `E${idx + 1}: F:${item.frame_id} (${timestampSec.toFixed(3)}s) — tua video tới sự kiện`;
            pill.setAttribute("aria-label", `Tua video tới sự kiện E${idx + 1} tại ${mins}:${secs}`);
            pill.addEventListener("click", () => {
                const isSameOpenVideo = videoModal?.style.display !== "none"
                    && currentPlayingVideoId === trakeTimelineState.videoId
                    && html5VideoPlayer;
                if (isSameOpenVideo) {
                    html5VideoPlayer.currentTime = timestampSec;
                    _updateLiveVideoBadge(true);
                    showToast(`⏱️ Tua về E${idx + 1}: ${mins}:${secs}`, "info", 1000);
                } else {
                    openVideoPlayer(trakeTimelineState.videoId, item.frame_id, timestampSec, 0);
                    showToast(`▶ Mở video tại E${idx + 1}: ${mins}:${secs}`, "info", 1200);
                }
            });
            trakeEventPills.appendChild(pill);
        });
    }

    if (trakeTimelineState.selectedFrames.length === 0) {
        if (trakeSelectedChainText) trakeSelectedChainText.innerText = "Chưa chọn frame nào (Ctrl+Click để chọn theo thứ tự E1, E2, E3...)";
        if (btnTrakeSubmitDres) {
            btnTrakeSubmitDres.disabled = true;
            btnTrakeSubmitDres.innerText = "🚀 NỘP CHUỖI TRAKE";
        }
    } else {
        if (trakeSelectedChainText) {
            trakeSelectedChainText.innerText = `${trakeTimelineState.videoId} · ${trakeTimelineState.selectedFrames.length} sự kiện`;
        }
        if (btnTrakeSubmitDres) {
            const count = trakeTimelineState.selectedFrames.length;
            btnTrakeSubmitDres.disabled = (count < 2);
            btnTrakeSubmitDres.innerText = count < 2
                ? `🚀 CẦN ÍT NHẤT 2 SỰ KIỆN (${count}/2)`
                : `🚀 NỘP CHUỖI TRAKE (${count} sự kiện • Ctrl+Enter)`;
        }
    }

    if (videoTrakeSubmitBtn) {
        const count = trakeTimelineState.selectedFrames.length;
        videoTrakeSubmitBtn.disabled = count < 2;
        videoTrakeSubmitBtn.innerText = count < 2
            ? `🚀 TRAKE (${count}/2)`
            : `🚀 NỘP CHUỖI TRAKE (${count})`;
    }
}

function toggleTrakeTimelineItem(f, itemDiv) {
    const existingIdx = trakeTimelineState.selectedFrames.findIndex(x => String(x.frame_id) === String(f.frame_id));
    if (existingIdx >= 0) {
        trakeTimelineState.selectedFrames.splice(existingIdx, 1);
        showToast(`↺ Đã bỏ chọn Frame ${f.frame_id}`, "info", 1200);
    } else {
        trakeTimelineState.selectedFrames.push({
            frame_id: f.frame_id,
            timestamp_sec: f.timestamp_sec,
            element: itemDiv
        });
        showToast(`✨ Đã thêm Frame ${f.frame_id} (E${trakeTimelineState.selectedFrames.length}) vào chuỗi TRAKE`, "success", 1200);
    }
    // TUYỆT ĐỐI KHÔNG sort theo timestamp để bảo toàn 100% thứ tự click của người dùng:
    // Sự kiện chọn đầu tiên luôn là E1, sự kiện tiếp theo là E2, E3...
    updateTrakeTimelineUi();
}

async function openTimeline(videoId, targetFrameId = null, activateTrake = false) {
    if (!videoId) return;

    if (timelineVideoTitle) timelineVideoTitle.innerText = `🎞️ TIMELINE: ${videoId}`;
    if (timelineFrameCount) timelineFrameCount.innerText = "Đang tải...";
    if (timelineStrip) timelineStrip.innerHTML = `<div style="padding: 30px; font-weight: bold; width: 100%; text-align: center;">⏳ Đang tải toàn bộ keyframes của video ${videoId}...</div>`;
    if (timelineModal) timelineModal.style.display = "flex";

    const isTrakeMode = activateTrake || (typeof dresTaskModeSelect !== "undefined" && dresTaskModeSelect && dresTaskModeSelect.value === "trake");
    const isSameTrakeVideo = trakeTimelineState.videoId === videoId;
    if (!isSameTrakeVideo) {
        trakeTimelineState.selectedFrames = [];
    } else {
        // Timeline DOM sẽ được dựng lại; giữ selection nhưng bỏ tham chiếu node cũ.
        trakeTimelineState.selectedFrames.forEach(frame => { frame.element = null; });
    }
    trakeTimelineState.videoId = videoId;

    if (trakeTimelineBar) {
        trakeTimelineBar.style.display = isTrakeMode ? "flex" : "none";
    }

    try {
        const resp = await fetch(`${TIMELINE_URL}/${videoId}`);
        if (!resp.ok) throw new Error(`Timeline Error: ${resp.statusText}`);

        const data = await resp.json();
        const frames = data.frames || [];

        if (timelineFrameCount) timelineFrameCount.innerText = `${frames.length} keyframes`;

        if (frames.length === 0) {
            timelineStrip.innerHTML = `<div style="padding: 30px; font-weight: bold; width: 100%; text-align: center;">Không tìm thấy keyframe nào của ${videoId}.</div>`;
            return;
        }

        timelineStrip.innerHTML = "";
        let targetElementToScroll = null;

        frames.forEach((f, fIdx) => {
            const itemDiv = document.createElement("div");
            itemDiv.className = "timeline-item";
            itemDiv.dataset.frameId = f.frame_id;

            // Nếu đây là frame được bấm mở từ thẻ kết quả -> Đánh dấu active và chuẩn bị cuộn tới
            if (targetFrameId && String(f.frame_id) === String(targetFrameId)) {
                itemDiv.classList.add("timeline-item-active");
                targetElementToScroll = itemDiv;
                // Lưu ý: Không tự động chèn targetFrameId vào selectedFrames để người dùng
                // toàn quyền chủ động chọn thứ tự E1, E2, E3... bằng Ctrl+Click
            }

            // Click vào thẻ keyframe:
            // - Nếu giữ Ctrl/Meta: Tuân theo đúng chế độ nộp hiện tại (KIS / QA / TRAKE)
            // - Nếu click thường: Mở Lightbox soi chi tiết
            itemDiv.addEventListener("click", (e) => {
                if (e.ctrlKey || e.metaKey) {
                    e.stopPropagation();
                    e.preventDefault();

                    const currentModeNow = (typeof dresTaskModeSelect !== "undefined" && dresTaskModeSelect)
                        ? dresTaskModeSelect.value
                        : "kis";

                    if (currentModeNow === "kis") {
                        // KIS MODE: Nộp trực tiếp lập tức lên DRES
                        showToast(`⚡ Đang nộp KIS từ Timeline: ${videoId} / Frame ${f.frame_id}`, "info", 1500);
                        itemDiv.style.outline = "2px solid #f59e0b";
                        handleDresFastSubmit(videoId, f.frame_id, itemDiv);
                        return;
                    }

                    if (currentModeNow === "qa") {
                        // QA MODE: Mở modal nhập đáp án QA cho frame này
                        handleDresFastSubmit(videoId, f.frame_id, itemDiv);
                        return;
                    }

                    if (currentModeNow === "trake") {
                        // TRAKE MODE: Chọn sự kiện E1, E2, E3...
                        if (trakeTimelineBar) trakeTimelineBar.style.display = "flex";
                        toggleTrakeTimelineItem(f, itemDiv);
                        return;
                    }
                }
                openLightbox(fIdx, frames);
            });

            const mins = Math.floor(f.timestamp_sec / 60);
            const secs = Math.floor(f.timestamp_sec % 60);
            const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            const asrIcon = f.asr_transcript ? `🎙️ ` : `⏱️ `;

            itemDiv.innerHTML = `
                <div class="timeline-thumb-box" title="🔍 Bấm để soi chi tiết | ⚡ Ctrl+Click: Chọn sự kiện TRAKE">
                    <img src="${f.image_url}" loading="lazy" alt="Frame ${f.frame_id}">
                    <span class="timeline-time-badge">${asrIcon}${timeStr}</span>
                </div>
                <div class="timeline-item-meta">
                    <span class="timeline-meta-frame">F:${f.frame_id}</span>
                    <span class="timeline-meta-sec">${f.timestamp_sec}s</span>
                </div>
            `;
            timelineStrip.appendChild(itemDiv);
        });

        if (isTrakeMode) {
            updateTrakeTimelineUi();
        }

        // TỰ ĐỘNG CUỘN ĐẾN ĐÚNG VỊ TRÍ FRAME ĐÃ CHỌN
        if (targetElementToScroll) {
            setTimeout(() => {
                targetElementToScroll.scrollIntoView({ behavior: "smooth", block: "center" });
            }, 80);
        }

    } catch (err) {
        if (timelineStrip) timelineStrip.innerHTML = `<div style="padding: 30px; color: red; font-weight: bold;">Lỗi tải timeline: ${err.message}</div>`;
        console.error(err);
    }
}

function closeTimeline() {
    if (timelineModal) timelineModal.style.display = "none";
}

if (timelineCloseBtn) timelineCloseBtn.addEventListener("click", closeTimeline);

if (timelineModal) {
    timelineModal.addEventListener("click", (e) => {
        if (e.target === timelineModal) closeTimeline();
    });
}

// Bắt sự kiện thanh công cụ TRAKE Timeline
const btnTrakeSortTime = document.getElementById("btn-trake-sort-time");
if (btnTrakeSortTime) {
    btnTrakeSortTime.addEventListener("click", () => {
        if (!trakeTimelineState.selectedFrames || trakeTimelineState.selectedFrames.length <= 1) {
            showToast("ℹ️ Cần chọn ít nhất 2 frame để sắp xếp theo thời gian!", "info", 1500);
            return;
        }
        trakeTimelineState.selectedFrames.sort((a, b) => (Number(a.timestamp_sec) || 0) - (Number(b.timestamp_sec) || 0));
        updateTrakeTimelineUi();
        showToast("⏱️ Đã sắp xếp lại các sự kiện theo thứ tự thời gian tăng dần!", "success", 1500);
    });
}

if (btnTrakeClearSelection) {
    btnTrakeClearSelection.addEventListener("click", () => {
        trakeTimelineState.selectedFrames = [];
        updateTrakeTimelineUi();
        showToast("↺ Đã làm mới chuỗi chọn TRAKE.", "info", 1500);
    });
}

if (btnTrakeSubmitDres) {
    btnTrakeSubmitDres.addEventListener("click", () => {
        if (!trakeTimelineState.videoId || trakeTimelineState.selectedFrames.length < 2) {
            showToast("⚠️ Cần chọn ít nhất 2 frame theo thứ tự thời gian để nộp TRAKE!", "warning");
            return;
        }
        const frameIds = trakeTimelineState.selectedFrames.map(f => f.frame_id);
        triggerDresTrakeSubmit(trakeTimelineState.videoId, frameIds, btnTrakeSubmitDres);
    });
}

if (videoTrakeSubmitBtn) {
    videoTrakeSubmitBtn.addEventListener("click", () => {
        if (!trakeTimelineState.videoId || trakeTimelineState.selectedFrames.length < 2) {
            showToast("⚠️ Cần chọn ít nhất 2 sự kiện TRAKE bằng Ctrl+Click vào video!", "warning");
            return;
        }
        const frameIds = trakeTimelineState.selectedFrames.map(frame => frame.frame_id);
        triggerDresTrakeSubmit(trakeTimelineState.videoId, frameIds, videoTrakeSubmitBtn);
    });
}

// Event Listeners cho Lightbox
if (lightboxCloseBtn) lightboxCloseBtn.addEventListener("click", closeLightbox);

if (lightboxToggleBoxes) {
    lightboxToggleBoxes.addEventListener("change", () => {
        drawLightboxCanvas();
    });
}
window.addEventListener("resize", () => {
    if (lightboxModal && lightboxModal.style.display !== "none") {
        drawLightboxCanvas();
    }
});

// Đóng lightbox khi click ra ngoài vùng modal
if (lightboxModal) {
    lightboxModal.addEventListener("click", (e) => {
        if (e.target === lightboxModal) closeLightbox();
    });
}

// Phím tắt bàn phím
window.addEventListener("keydown", (e) => {
    const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
    if (activeTag === "input" || activeTag === "textarea") return;

    if (dresQaModal && dresQaModal.style.display !== "none") {
        if (e.key === "Escape") {
            e.preventDefault();
            dresQaModal.style.display = "none";
            return;
        }
    }

    if (videoModal && videoModal.style.display !== "none") {
        if (e.key === "Escape") {
            closeVideoPlayer();
        }
    } else if (lightboxModal && lightboxModal.style.display !== "none") {
        if (e.key === "Escape") {
            closeLightbox();
        } else if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S" || e.key === "Enter")) {
            e.preventDefault();
            if (activeResultsSource && currentLightboxIndex >= 0 && currentLightboxIndex < activeResultsSource.length) {
                const item = activeResultsSource[currentLightboxIndex];
                const bestFid = item.best_kf_frame_id || item.anchor_frame_id || item.frame_id || 0;
                handleDresFastSubmit(item.video_id, bestFid, lightboxDresBtn);
            }
        } else if (e.key === "ArrowLeft") {
            e.preventDefault();
            if (currentLightboxIndex > 0) {
                openLightbox(currentLightboxIndex - 1, activeResultsSource);
            }
        } else if (e.key === "ArrowRight") {
            e.preventDefault();
            if (activeResultsSource && currentLightboxIndex < activeResultsSource.length - 1) {
                openLightbox(currentLightboxIndex + 1, activeResultsSource);
            }
        }
    } else if (timelineModal && timelineModal.style.display !== "none") {
        if (e.key === "Escape") {
            closeTimeline();
        } else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            if (btnTrakeSubmitDres && !btnTrakeSubmitDres.disabled) {
                btnTrakeSubmitDres.click();
            }
        }
    }
});

// Visual Query có handler riêng để phân biệt truy vấn đơn cảnh và TRAKE nhiều đoạn.
[ocrInput, speechInput, videoIdInput, topKNum, visualSimNum].forEach(el => {
    if (el) {
        el.addEventListener("keypress", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                performSearch();
            }
        });
    }
});

if (multiSearchBtn) multiSearchBtn.addEventListener("click", performMultiSceneSearch);
if (trakeSearchBtn) trakeSearchBtn.addEventListener("click", performTrakeSearch);
if (trakeExportBtn) trakeExportBtn.addEventListener("click", exportTrakeCsv);

// Tự động lưu trạng thái khi người dùng thay đổi bất kỳ tham số nào
[
    objectClassSelect, objectConfSlider,
    weightVisualSlider, weightOcrSlider, weightAsrSlider, weightObjectSlider,
    visualSimSlider, visualSimNum, uniqueVideoCheck,
    topKSlider, topKNum, rerankTopV, rerankKPerVideo,
    trakeLambdaInput, trakeKpathsInput, trakeTopKInput
].forEach(el => {
    if (el) {
        el.addEventListener("input", saveControlPanelState);
        el.addEventListener("change", saveControlPanelState);
    }
});

document.querySelectorAll('input[name="smart-submode"]').forEach(r => {
    r.addEventListener("change", saveControlPanelState);
});

// Khôi phục trạng thái bảng điều khiển từ localStorage khi khởi động
restoreControlPanelState();

// OCR đang ẩn: vô hiệu hóa cả state cũ trong localStorage để không âm thầm
// tác động tới truy vấn trong khi người vận hành không nhìn thấy module này.
if (!OCR_FEATURE_VISIBLE && enableOcrCheck) {
    enableOcrCheck.checked = false;
    enableOcrCheck.disabled = true;
}

// ==========================================================================
// SUBMISSION MANAGER LOGIC & HANDLERS (AIC 2026 CONTEST SUITE)
// ==========================================================================

function switchSubMode(mode) {
    currentSubMode = mode;
    if (subModePills) {
        subModePills.querySelectorAll(".btn-pill").forEach(btn => {
            if (btn.getAttribute("data-mode") === mode) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    }

    if (mode === "kis") {
        if (subKisQaFields) subKisQaFields.style.display = "block";
        if (subQaFields) subQaFields.style.display = "none";
        if (subTrakeFields) subTrakeFields.style.display = "none";
        if (subHedgingRow) subHedgingRow.style.display = "block";
    } else if (mode === "qa") {
        if (subKisQaFields) subKisQaFields.style.display = "block";
        if (subQaFields) subQaFields.style.display = "block";
        if (subTrakeFields) subTrakeFields.style.display = "none";
        if (subHedgingRow) subHedgingRow.style.display = "block";
    } else if (mode === "trake") {
        if (subKisQaFields) subKisQaFields.style.display = "none";
        if (subQaFields) subQaFields.style.display = "none";
        if (subTrakeFields) subTrakeFields.style.display = "block";
        if (subHedgingRow) subHedgingRow.style.display = "none";
    }
}

if (subModePills) {
    subModePills.querySelectorAll(".btn-pill").forEach(btn => {
        btn.addEventListener("click", () => {
            const mode = btn.getAttribute("data-mode") || "kis";
            if (typeof setDresSubmissionMode === "function") {
                setDresSubmissionMode(mode);
            } else {
                switchSubMode(mode);
            }
        });
    });
}

const subHedgingRow = document.getElementById("sub-hedging-row");

async function loadSubmissionDirectory() {
    const dir = subDirPathInput ? subDirPathInput.value.trim() : "query/query-p1/submission";
    if (!dir) return;

    try {
        const response = await fetch(SUBMISSION_LIST_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dir_path: dir })
        });

        if (!response.ok) return;

        const data = await response.json();
        if (data.status === "success" && data.exists && subFileListContainer && subFileItemsContainer) {
            subFileListContainer.style.display = "block";
            subFileCountSpan.innerText = data.file_count;
            subFileItemsContainer.innerHTML = "";

            data.files.forEach(f => {
                const row = document.createElement("div");
                row.className = "sub-file-item-row";
                const typeClass = f.type === "QA" ? "sub-type-qa" : (f.type === "TRAKE" ? "sub-type-trake" : "sub-type-kis");
                row.innerHTML = `
                    <span style="font-weight: 600;">${f.filename}</span>
                    <div style="display: flex; align-items: center; gap: 4px;">
                        <span class="sub-type-badge ${typeClass}">${f.type}</span>
                        <span style="color: #94a3b8;">${f.line_count}d</span>
                    </div>
                `;
                row.addEventListener("click", () => {
                    if (subFileNameInput) subFileNameInput.value = f.filename;
                    showToast(`📄 Đã chọn file: ${f.filename}`, "info", 1500);
                });
                subFileItemsContainer.appendChild(row);
            });
        }
    } catch (e) {
        console.warn("Could not load submission directory:", e);
    }
}

if (btnSubRefreshDir) {
    btnSubRefreshDir.addEventListener("click", () => {
        loadSubmissionDirectory();
        showToast("🔄 Đã quét lại thư mục submission", "info", 1500);
    });
}

if (subDirPathInput) {
    subDirPathInput.addEventListener("change", () => {
        saveControlPanelState();
        loadSubmissionDirectory();
    });
}

async function performSubmissionFill() {
    const vId = subVideoIdInput ? subVideoIdInput.value.trim() : "";
    if (!vId && currentSubMode !== "trake") {
        showToast("⚠️ Vui lòng nhập Video ID (Ví dụ: L21_V015)", "warning");
        return;
    }

    const fA = subFrameAInput ? parseInt(subFrameAInput.value) : null;
    const fB = subFrameBInput ? parseInt(subFrameBInput.value) : null;
    const fA2 = subFrameA2Input ? parseInt(subFrameA2Input.value) : null;
    const fB2 = subFrameB2Input ? parseInt(subFrameB2Input.value) : null;
    const answer = subQaAnswerInput ? subQaAnswerInput.value.trim() : "";

    let secondInterval = null;
    if (!isNaN(fA2) && !isNaN(fB2) && fA2 !== null && fB2 !== null) {
        secondInterval = [fA2, fB2];
    }

    let trakeFrames = null;
    if (currentSubMode === "trake") {
        const rawT = subTrakeFramesInput ? subTrakeFramesInput.value.trim() : "";
        if (!rawT) {
            showToast("⚠️ Vui lòng nhập chuỗi Frame IDs cho các sự kiện TRAKE", "warning");
            return;
        }
        trakeFrames = rawT.split(/[\s,]+/).filter(x => x).map(x => parseInt(x)).filter(x => !isNaN(x));
        if (trakeFrames.length < 2) {
            showToast("⚠️ Chuỗi TRAKE phải chứa ít nhất 2 Frame IDs", "warning");
            return;
        }
    }

    // Thu thập System Backup Items từ kết quả tìm kiếm hiện tại nếu bật Risk Hedging
    let backupItems = null;
    if (subEnableHedgingCheck && subEnableHedgingCheck.checked && currentResults && currentResults.length > 0) {
        backupItems = currentResults
            .filter(r => r.video_id !== vId)
            .map(r => ({ video_id: r.video_id, frame_id: r.frame_id }));
    }

    if (btnSubFillPreview) {
        btnSubFillPreview.disabled = true;
        btnSubFillPreview.innerText = "⏳ ĐANG TÍNH...";
    }

    try {
        const response = await fetch(SUBMISSION_FILL_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                mode: currentSubMode,
                video_id: vId,
                frame_a: fA,
                frame_b: fB,
                second_interval: secondInterval,
                answer: answer,
                trake_event_frames: trakeFrames,
                system_backup_items: backupItems,
                target_count: 100
            })
        });

        if (!response.ok) {
            throw new Error(`Fill Error: ${response.statusText}`);
        }

        const data = await response.json();
        if (data.status === "success") {
            latestGeneratedSubmissionLines = data.lines || [];
            if (subPreviewContainer && subPreviewCode && subPreviewTotalSpan) {
                subPreviewContainer.style.display = "block";
                subPreviewTotalSpan.innerText = data.total_lines;
                subPreviewCode.innerText = (data.preview || []).join("\n") + (data.total_lines > 5 ? `\n... (+${data.total_lines - 5} dòng khác)` : "");
                if (subPreviewStatusSpan) {
                    subPreviewStatusSpan.innerText = `✅ Đã sinh ${data.total_lines} dòng (${data.mode.toUpperCase()})`;
                }
            }
            showToast(`🎯 Đã sinh thành công ${data.total_lines} dòng submission theo chiến thuật!`, "success");
        } else {
            showToast(`Lỗi: ${data.message}`, "error");
        }
    } catch (err) {
        showToast(`Lỗi fill submission: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (btnSubFillPreview) {
            btnSubFillPreview.disabled = false;
            btnSubFillPreview.innerText = "🎯 FILL & XEM";
        }
    }
}

if (btnSubFillPreview) {
    btnSubFillPreview.addEventListener("click", performSubmissionFill);
}

async function saveSubmissionFile() {
    if (!latestGeneratedSubmissionLines || latestGeneratedSubmissionLines.length === 0) {
        showToast("⚠️ Chưa có nội dung. Vui lòng bấm '🎯 FILL & XEM' trước khi lưu.", "warning");
        return;
    }

    let fname = subFileNameInput ? subFileNameInput.value.trim() : "";
    if (!fname) {
        showToast("⚠️ Vui lòng nhập tên file CSV (Ví dụ: query-p1-1-kis.csv)", "warning");
        return;
    }
    if (!fname.toLowerCase().endsWith(".csv")) {
        fname += ".csv";
        if (subFileNameInput) subFileNameInput.value = fname;
    }

    const dir = subDirPathInput ? subDirPathInput.value.trim() : "query/query-p1/submission";
    const fullPath = `${dir}/${fname}`.replace(/\/+/g, "/");

    if (btnSubSaveFile) {
        btnSubSaveFile.disabled = true;
        btnSubSaveFile.innerText = "⏳ ĐANG LƯU...";
    }

    try {
        const response = await fetch(SUBMISSION_SAVE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                file_path: fullPath,
                lines: latestGeneratedSubmissionLines
            })
        });

        if (!response.ok) throw new Error(response.statusText);

        const data = await response.json();
        if (data.status === "success") {
            showToast(`💾 Đã lưu thành công ${data.line_count} dòng vào ${fname}!`, "success", 3000);
            loadSubmissionDirectory();
        } else {
            showToast(`Lỗi: ${data.message}`, "error");
        }
    } catch (err) {
        showToast(`Lỗi lưu file: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (btnSubSaveFile) {
            btnSubSaveFile.disabled = false;
            btnSubSaveFile.innerText = "💾 LƯU FILE CSV";
        }
    }
}

if (btnSubSaveFile) {
    btnSubSaveFile.addEventListener("click", saveSubmissionFile);
}

async function exportSubmissionZip() {
    const dir = subDirPathInput ? subDirPathInput.value.trim() : "query/query-p1/submission";
    let zipOut = subZipOutputPathInput ? subZipOutputPathInput.value.trim() : "query/submission_p1.zip";
    if (zipOut && !zipOut.toLowerCase().endsWith(".zip")) {
        zipOut += ".zip";
        if (subZipOutputPathInput) subZipOutputPathInput.value = zipOut;
    }

    if (btnSubExportZip) {
        btnSubExportZip.disabled = true;
        btnSubExportZip.innerText = "⏳ ĐANG ĐÓNG GÓI ZIP...";
    }

    try {
        const response = await fetch(SUBMISSION_ZIP_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                submission_dir: dir,
                output_zip_path: zipOut
            })
        });

        if (!response.ok) throw new Error(response.statusText);

        const data = await response.json();
        if (data.status === "success") {
            showToast(`📦 ĐÃ XUẤT THÀNH CÔNG ${data.file_count} FILES VÀO ${zipOut} (${data.file_size_kb} KB)!`, "success", 4000);
        } else {
            showToast(`Lỗi: ${data.message}`, "error");
        }
    } catch (err) {
        showToast(`Lỗi xuất zip: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (btnSubExportZip) {
            btnSubExportZip.disabled = false;
            btnSubExportZip.innerText = "📦 XUẤT SUBMISSION.ZIP (CHUẨN BTC)";
        }
    }
}

if (btnSubExportZip) {
    btnSubExportZip.addEventListener("click", exportSubmissionZip);
}

// Nạp nhanh Keyframe từ Lightbox vào Submission Manager
const lightboxDresBtn = document.getElementById("lightbox-dres-submit-btn");
if (lightboxDresBtn) {
    lightboxDresBtn.addEventListener("click", () => {
        if (!activeResultsSource || currentLightboxIndex < 0 || currentLightboxIndex >= activeResultsSource.length) return;
        const item = activeResultsSource[currentLightboxIndex];
        const bestFid = item.best_kf_frame_id || item.anchor_frame_id || item.frame_id || 0;
        handleDresFastSubmit(item.video_id, bestFid, lightboxDresBtn);
    });
}

if (lightboxSubBtn) {
    lightboxSubBtn.addEventListener("click", () => {
        const item = activeResultsSource[currentLightboxIndex];
        if (!item) return;

        let fid = item.best_kf_frame_id || item.anchor_frame_id || item.frame_id || "";
        if (typeof fid === "number") fid = String(fid).padStart(6, "0");

        if (subVideoIdInput) subVideoIdInput.value = item.video_id;
        if (subFrameAInput) subFrameAInput.value = fid;
        if (subFrameBInput && !subFrameBInput.value && fid) {
            const numFid = parseInt(fid);
            if (!isNaN(numFid)) {
                subFrameBInput.value = String(numFid + 300).padStart(6, "0");
            }
        }

        // Mở card Submission nếu đang bị thu gọn
        if (cardSubmission && cardSubmission.classList.contains("is-collapsed")) {
            setCardCollapsed(cardSubmission, false);
        }

        // Cuộn tới Card Submission
        if (cardSubmission) {
            cardSubmission.scrollIntoView({ behavior: "smooth", block: "center" });
        }

        showToast(`📥 Đã nạp ${item.video_id} frame ${fid} vào Submission Manager!`, "success", 2000);
    });
}

// ==========================================================================
// MULTI-VIEW IMAGE GALLERY & CLIPBOARD PASTE LISTENER
// ==========================================================================
let currentImageQueries = []; // Danh sách các ảnh base64 đa góc nhìn

function addImageQuery(base64Data) {
    if (!base64Data) return;
    currentImageQueries.push(base64Data);
    currentImageQueryBase64 = currentImageQueries[0];
    renderImageGallery();
    if (cardImageQuery && cardImageQuery.classList.contains("is-collapsed")) {
        setCardCollapsed(cardImageQuery, false);
    }
    showToast(`📸 Đã nạp góc ảnh #${currentImageQueries.length} thành công!`, "info");
}

function removeImageQuery(idx) {
    if (idx >= 0 && idx < currentImageQueries.length) {
        currentImageQueries.splice(idx, 1);
        currentImageQueryBase64 = currentImageQueries.length > 0 ? currentImageQueries[0] : null;
        renderImageGallery();
        showToast("🗑️ Đã xóa góc ảnh.", "info");
    }
}

function clearAllImageQueries() {
    currentImageQueries = [];
    currentImageQueryBase64 = null;
    renderImageGallery();
    if (imageFileInput) imageFileInput.value = "";
    showToast("🗑️ Đã xóa toàn bộ ảnh.", "info");
}

function renderImageGallery() {
    const gallery = document.getElementById("image-gallery-container");
    const placeholder = document.getElementById("image-dropzone-placeholder");
    const actionsBar = document.getElementById("image-actions-bar");
    const badgeCount = document.getElementById("badge-image-count");

    if (!gallery || !placeholder) return;

    gallery.innerHTML = "";
    if (currentImageQueries.length === 0) {
        gallery.style.display = "none";
        placeholder.style.display = "block";
        if (actionsBar) actionsBar.style.display = "none";
        if (badgeCount) badgeCount.style.display = "none";
    } else {
        gallery.style.display = "grid";
        placeholder.style.display = "none";
        if (actionsBar) actionsBar.style.display = "flex";
        if (badgeCount) {
            badgeCount.style.display = "inline-block";
            badgeCount.innerText = `${currentImageQueries.length} ảnh`;
        }

        currentImageQueries.forEach((b64, idx) => {
            const thumbWrap = document.createElement("div");
            thumbWrap.style.cssText = "position: relative; border-radius: 6px; border: 1px solid #6366f1; background: #0f172a; overflow: hidden; height: 80px; display: flex; align-items: center; justify-content: center;";
            
            thumbWrap.innerHTML = `
                <img src="${b64}" style="max-width: 100%; max-height: 100%; object-fit: cover;" alt="Góc ảnh #${idx + 1}">
                <span style="position: absolute; bottom: 2px; left: 2px; background: rgba(0,0,0,0.75); color: #fff; font-size: 0.65rem; padding: 1px 4px; border-radius: 3px; font-weight: 700;">#${idx + 1}</span>
                <button type="button" onclick="event.stopPropagation(); removeImageQuery(${idx})" style="position: absolute; top: 2px; right: 2px; background: rgba(239, 68, 68, 0.9); color: white; border: none; border-radius: 50%; width: 18px; height: 18px; cursor: pointer; font-size: 0.65rem; line-height: 1; display: flex; align-items: center; justify-content: center;" title="Xóa góc ảnh này">✕</button>
            `;
            gallery.appendChild(thumbWrap);
        });
    }
}
window.removeImageQuery = removeImageQuery;
window.clearAllImageQueries = clearAllImageQueries;

// Global Clipboard Paste Listener (Ctrl + V)
window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData))?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file' && item.type.startsWith('image/')) {
            const blob = item.getAsFile();
            const reader = new FileReader();
            reader.onload = (event) => {
                addImageQuery(event.target.result);
            };
            reader.readAsDataURL(blob);
        }
    }
});

// Dropzone file selection & drag-and-drop
if (imageDropzone) {
    imageDropzone.addEventListener("click", (e) => {
        if (e.target.tagName === "BUTTON") return;
        if (imageFileInput) imageFileInput.click();
    });

    imageDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        imageDropzone.style.borderColor = "#6366f1";
        imageDropzone.style.background = "rgba(99, 102, 241, 0.15)";
    });

    imageDropzone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        imageDropzone.style.borderColor = "#475569";
        imageDropzone.style.background = "rgba(15, 23, 42, 0.5)";
    });

    imageDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        imageDropzone.style.borderColor = "#475569";
        imageDropzone.style.background = "rgba(15, 23, 42, 0.5)";
        if (e.dataTransfer && e.dataTransfer.files) {
            Array.from(e.dataTransfer.files).forEach(file => {
                if (file.type.startsWith("image/")) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        addImageQuery(event.target.result);
                    };
                    reader.readAsDataURL(file);
                }
            });
        }
    });
}

if (imageFileInput) {
    imageFileInput.addEventListener("change", (e) => {
        if (e.target.files) {
            Array.from(e.target.files).forEach(file => {
                const reader = new FileReader();
                reader.onload = (event) => {
                    addImageQuery(event.target.result);
                };
                reader.readAsDataURL(file);
            });
        }
    });
}

const btnAddMoreImage = document.getElementById("btn-add-more-image");
if (btnAddMoreImage) {
    btnAddMoreImage.addEventListener("click", (e) => {
        e.stopPropagation();
        if (imageFileInput) imageFileInput.click();
    });
}

const btnClearAllImages = document.getElementById("btn-clear-all-images");
if (btnClearAllImages) {
    btnClearAllImages.addEventListener("click", (e) => {
        e.stopPropagation();
        clearAllImageQueries();
    });
}

// Polymorphic Image Search Action (Kế thừa và đa hình 100% từ performSearch)
async function executeImageSearch() {
    return performSearch();
}

if (btnSearchImageQuery) {
    btnSearchImageQuery.addEventListener("click", executeImageSearch);
}
window.executeImageSearch = executeImageSearch;

// Tag UI Helper Functions (Hỗ trợ thêm nhiều video ID / Prefix đồng thời)
function addIncludeTokens(rawText) {
    if (!rawText) return;
    const tokens = rawText.split(/[,;\s\n\t]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
    let changed = false;
    tokens.forEach(tok => {
        if (!currentIncludeTags.includes(tok)) {
            currentIncludeTags.push(tok);
            changed = true;
        }
    });
    if (changed) renderIncludeTags();
}

function addExcludeTokens(rawText) {
    if (!rawText) return;
    const tokens = rawText.split(/[,;\s\n\t]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
    let changed = false;
    tokens.forEach(tok => {
        if (!currentExcludeTags.includes(tok)) {
            currentExcludeTags.push(tok);
            changed = true;
        }
    });
    if (changed) renderExcludeTags();
}

function commitPendingTags() {
    if (includeVideoInput && includeVideoInput.value.trim()) {
        addIncludeTokens(includeVideoInput.value);
        includeVideoInput.value = "";
    }
    if (excludeVideoInput && excludeVideoInput.value.trim()) {
        addExcludeTokens(excludeVideoInput.value);
        excludeVideoInput.value = "";
    }
}
window.commitPendingTags = commitPendingTags;

function renderIncludeTags() {
    if (!includeTagsContainer) return;
    includeTagsContainer.innerHTML = "";
    currentIncludeTags.forEach((tag, idx) => {
        const span = document.createElement("span");
        span.className = "tag-item tag-include";
        span.innerHTML = `<span>${tag}</span><span class="tag-remove-btn" title="Xóa" onclick="removeIncludeTag(${idx})">&times;</span>`;
        includeTagsContainer.appendChild(span);
    });
}

function removeIncludeTag(idx) {
    currentIncludeTags.splice(idx, 1);
    renderIncludeTags();
}
window.removeIncludeTag = removeIncludeTag;

function renderExcludeTags() {
    if (!excludeTagsContainer) return;
    excludeTagsContainer.innerHTML = "";
    currentExcludeTags.forEach((tag, idx) => {
        const span = document.createElement("span");
        span.className = "tag-item tag-exclude";
        span.innerHTML = `<span>${tag}</span><span class="tag-remove-btn" title="Xóa" onclick="removeExcludeTag(${idx})">&times;</span>`;
        excludeTagsContainer.appendChild(span);
    });
}

function removeExcludeTag(idx) {
    currentExcludeTags.splice(idx, 1);
    renderExcludeTags();
    // Đồng bộ trạng thái mờ/sáng trên lưới kết quả
    applyExcludeStateToGrid();
}
window.removeExcludeTag = removeExcludeTag;

if (includeVideoInput) {
    includeVideoInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            addIncludeTokens(includeVideoInput.value);
            includeVideoInput.value = "";
        }
    });
    includeVideoInput.addEventListener("paste", () => {
        setTimeout(() => {
            addIncludeTokens(includeVideoInput.value);
            includeVideoInput.value = "";
        }, 30);
    });
}

if (excludeVideoInput) {
    excludeVideoInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            addExcludeTokens(excludeVideoInput.value);
            excludeVideoInput.value = "";
        }
    });
    excludeVideoInput.addEventListener("paste", () => {
        setTimeout(() => {
            addExcludeTokens(excludeVideoInput.value);
            excludeVideoInput.value = "";
        }, 30);
    });
}

// Client-side quick filter on Top K (Lọc an toàn từ currentResults gốc, hoàn tác 100% khi xóa bộ lọc)
if (filterVideoResultsInput) {
    filterVideoResultsInput.addEventListener("input", () => {
        const rawFilter = filterVideoResultsInput.value.trim().toLowerCase();
        if (!currentResults || currentResults.length === 0) return;
        
        if (!rawFilter) {
            resultCount.innerText = `${currentResults.length} items found`;
            renderResults(currentResults);
            return;
        }

        const tokens = rawFilter.split(/[,;\s]+/).map(t => t.trim()).filter(Boolean);
        if (tokens.length === 0) {
            resultCount.innerText = `${currentResults.length} items found`;
            renderResults(currentResults);
            return;
        }

        const posTokens = tokens.filter(t => !t.startsWith("-"));
        const negTokens = tokens.filter(t => t.startsWith("-")).map(t => t.slice(1)).filter(Boolean);

        const filtered = currentResults.filter(item => {
            const vid = (item.video_id || "").toLowerCase();
            const fid = String(item.frame_id || "").toLowerCase();
            const fullId = `${vid}_${fid}`;

            // Kiểm tra loại trừ
            for (const nt of negTokens) {
                if (vid.includes(nt) || fid.includes(nt) || fullId.includes(nt)) {
                    return false;
                }
            }

            if (posTokens.length === 0) return true;

            // Kiểm tra bao gồm (khớp ít nhất 1 trong các posTokens)
            return posTokens.some(pt => vid.includes(pt) || fid.includes(pt) || fullId.includes(pt));
        });

        resultCount.innerText = `${filtered.length} / ${currentResults.length} items (đang lọc)`;
        renderResults(filtered);
    });
}

// Nạp danh sách Danh Mục Chủ Đề từ Backend
let categoriesLoading = false;
let categoriesRetryTimer = null;
async function loadCategories(attempt = 0) {
    if (categoriesLoading) return;
    categoriesLoading = true;
    clearTimeout(categoriesRetryTimer);
    const status = document.getElementById("category-load-status");
    const retry = document.getElementById("category-load-retry");
    if (status) status.textContent = "Đang tải danh mục…";
    if (retry) retry.hidden = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
        const res = await fetch(CATEGORIES_URL, { signal: controller.signal, cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.status !== "success" || !data.taxonomy || !Object.keys(data.taxonomy).length) {
            throw new Error("API không trả danh mục hợp lệ");
        }
        if (data.status === "success" && data.taxonomy) {
            categoriesTaxonomy = data.taxonomy;
            if (categorySelect) {
                categorySelect.innerHTML = "";
                for (const [catKey, catInfo] of Object.entries(categoriesTaxonomy)) {
                    const opt = document.createElement("option");
                    opt.value = catKey;
                    const count = Number.isFinite(Number(catInfo.count)) ? ` · ${catInfo.count}` : "";
                    opt.textContent = `${catInfo.label || catKey}${count}`;
                    categorySelect.appendChild(opt);
                }
                applySelectedValues(categorySelect, pendingCategorySelection);
                refreshSubcategoryOptions();
            }
        }
        if (status) status.textContent = "";
    } catch (e) {
        console.warn("Could not load categories", e);
        if (status) status.textContent = attempt < 3
            ? "Chưa tải được danh mục — đang thử lại…"
            : "Không tải được danh mục. Kiểm tra backend rồi bấm Thử lại.";
        if (attempt < 3) categoriesRetryTimer = setTimeout(() => loadCategories(attempt + 1), 2000 * (attempt + 1));
        else if (retry) retry.hidden = false;
    } finally {
        clearTimeout(timeout);
        categoriesLoading = false;
    }
}
document.getElementById("category-load-retry")?.addEventListener("click", () => loadCategories());

function refreshSubcategoryOptions() {
    const selectedCategories = getSelectedValues(categorySelect);
    const teachingInfo = categoriesTaxonomy.day_hoc;
    const showSubjects = selectedCategories.includes("day_hoc") && teachingInfo?.sub;
    if (groupSubcategory) groupSubcategory.style.display = showSubjects ? "block" : "none";
    if (!subcategorySelect) return;

    const previous = getSelectedValues(subcategorySelect).length > 0
        ? getSelectedValues(subcategorySelect)
        : pendingSubcategorySelection;
    subcategorySelect.innerHTML = "";
    if (showSubjects) {
        for (const [subKey, subLabel] of Object.entries(teachingInfo.sub)) {
            const opt = document.createElement("option");
            opt.value = subKey;
            const count = teachingInfo.sub_counts?.[subKey];
            opt.textContent = `${subLabel}${count !== undefined ? ` · ${count}` : ""}`;
            subcategorySelect.appendChild(opt);
        }
        applySelectedValues(subcategorySelect, previous);
    }
}

if (categorySelect) {
    categorySelect.addEventListener("change", () => {
        pendingCategorySelection = getSelectedValues(categorySelect);
        refreshSubcategoryOptions();
        saveControlPanelState();
    });
}

if (subcategorySelect) {
    subcategorySelect.addEventListener("change", () => {
        pendingSubcategorySelection = getSelectedValues(subcategorySelect);
        saveControlPanelState();
    });
}

if (clearCategoryFilterBtn) {
    clearCategoryFilterBtn.addEventListener("click", () => {
        pendingCategorySelection = [];
        pendingSubcategorySelection = [];
        applySelectedValues(categorySelect, []);
        applySelectedValues(subcategorySelect, []);
        refreshSubcategoryOptions();
        saveControlPanelState();
        showToast("Đã xóa bộ lọc danh mục.", "info", 1500);
    });
}

// Nạp danh sách 80 lớp COCO cho Object Filter Dropdown
async function loadCocoClasses() {
    try {
        const res = await fetch(COCO_URL);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "success" && data.classes && objectClassSelect) {
            const currentVal = objectClassSelect.value;
            objectClassSelect.innerHTML = "";
            data.classes.forEach((clsName, idx) => {
                const opt = document.createElement("option");
                opt.value = idx;
                opt.innerText = `${idx}: ${clsName}`;
                objectClassSelect.appendChild(opt);
            });
            if (currentVal !== undefined && currentVal !== "") {
                objectClassSelect.value = currentVal;
            }
        }
    } catch (e) {
        console.warn("Could not load COCO classes", e);
    }
}

// Tự động nạp ban đầu khi khởi động web
setTimeout(loadSubmissionDirectory, 800);
setTimeout(loadCocoClasses, 300);
setTimeout(loadCategories, 400);







// ==============================================================================
// DRES LIVE CONTEST SUITE CONTROLLER (VÒNG CHUNG KẾT AIC 2026)
// ==============================================================================
const DRES_STATUS_URL = "/api/dres/status";
const DRES_LOGIN_URL = "/api/dres/login";
const DRES_REFRESH_URL = "/api/dres/refresh-eval";
const DRES_SET_EVAL_URL = "/api/dres/set-evaluation";
const DRES_SUBMIT_KIS_URL = "/api/dres/submit-kis";
const DRES_SUBMIT_QA_URL = "/api/dres/submit-qa";
const DRES_SUBMIT_TRAKE_URL = "/api/dres/submit-trake";

async function parseDresApiResponse(response) {
    const rawBody = await response.text();
    let payload = null;
    if (rawBody) {
        try {
            payload = JSON.parse(rawBody);
        } catch {
            payload = null;
        }
    }

    if (!response.ok) {
        const backendMessage = payload?.message || payload?.detail;
        throw new Error(
            backendMessage || `Backend DRES lỗi HTTP ${response.status}. Đây là lỗi kỹ thuật, chưa xác định đáp án đúng hay sai.`
        );
    }
    if (!payload || typeof payload !== "object") {
        throw new Error("Backend DRES trả dữ liệu không hợp lệ. Chưa xác định đáp án đúng hay sai.");
    }
    return payload;
}
const DRES_START_TASK_URL = "/api/dres/start-task";
const DRES_RUN_ID_STORAGE_KEY = "aic_2026_dres_run_id";

let dresState = {
    isLoggedIn: false,
    isConnected: false,
    username: localStorage.getItem("dres_username") || "team_297",
    serverUrl: localStorage.getItem("dres_server_url") || "https://eventretrieval.one",
    evaluationName: "",
    evaluationId: localStorage.getItem(DRES_RUN_ID_STORAGE_KEY) || "",
    wrongCount: 0,
    submittedKeys: new Set()
};

let pendingQaTarget = null; // { videoId, frameId, targetEl }

// DOM elements
const dresDot = document.getElementById("dres-dot");
const dresStatusLabel = document.getElementById("dres-status-label");
const dresEvalName = document.getElementById("dres-eval-name");
const dresRefreshEvalBtn = document.getElementById("dres-refresh-eval-btn");
const dresTaskModeSelect = document.getElementById("dres-task-mode-select");
const dresHistoryToggleBtn = document.getElementById("dres-history-toggle-btn");
const dresSubCount = document.getElementById("dres-sub-count");
const dresSettingsBtn = document.getElementById("dres-settings-btn");
const dresShortcutDesc = document.getElementById("dres-shortcut-desc");
const dresModePills = document.querySelectorAll("#dres-mode-pills .btn-pill");

const dresSettingsModal = document.getElementById("dres-settings-modal");
const dresSettingsClose = document.getElementById("dres-settings-close");
const dresInputServer = document.getElementById("dres-input-server");
const dresInputUser = document.getElementById("dres-input-user");
const dresInputPass = document.getElementById("dres-input-pass");
const dresInputEvalId = document.getElementById("dres-input-eval-id");
const dresCopyRunIdBtn = document.getElementById("dres-copy-run-id");
const dresBtnSaveLogin = document.getElementById("dres-btn-save-login");
let applyingSavedRunId = false;

function isUuidLike(value) {
    return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(String(value || "").trim());
}

async function setDresRunId(runId, { persist = true, quiet = false } = {}) {
    const cleanId = String(runId || "").trim();
    if (!cleanId) return { status: "error", message: "Run ID không được để trống." };
    if (!isUuidLike(cleanId)) return { status: "error", message: "Run ID không đúng định dạng UUID." };

    const response = await fetch(DRES_SET_EVAL_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evaluation_id: cleanId })
    });
    const data = await parseDresApiResponse(response);
    if (data.status === "success") {
        dresState.evaluationId = cleanId;
        if (persist) localStorage.setItem(DRES_RUN_ID_STORAGE_KEY, cleanId);
        if (!quiet) showToast(`Đã lưu Run ID ${cleanId.slice(0, 8)}…`, "success", 2200);
    }
    return data;
}

const dresHistoryModal = document.getElementById("dres-history-modal");
const dresHistoryClose = document.getElementById("dres-history-close");
const dresHistoryList = document.getElementById("dres-history-list");

const dresQaModal = document.getElementById("dres-qa-modal");
const dresQaClose = document.getElementById("dres-qa-close");
const qaModalTarget = document.getElementById("qa-modal-target");
const dresQaAnswerInput = document.getElementById("dres-qa-answer-input");
const dresQaPreviewText = document.getElementById("dres-qa-preview-text");
const dresQaSubmitBtn = document.getElementById("dres-qa-submit-btn");

// HÀM ĐỒNG BỘ TOÀN DIỆN CHẾ ĐỘ THI ĐẤU (KIS / QA / TRAKE)
function setDresSubmissionMode(mode) {
    const m = (mode || "kis").toLowerCase();

    // 1. Cập nhật select ẩn dresTaskModeSelect
    if (dresTaskModeSelect) dresTaskModeSelect.value = m;

    // 2. Cập nhật pills trên thanh DRES Live Bar
    if (dresModePills) {
        dresModePills.forEach(p => {
            p.classList.toggle("active", p.dataset.mode === m);
        });
    }

    // 3. Cập nhật pills trong Submission Manager (sidebar)
    if (subModePills) {
        subModePills.querySelectorAll(".btn-pill").forEach(p => {
            p.classList.toggle("active", p.getAttribute("data-mode") === m);
        });
    }
    if (typeof switchSubMode === "function") {
        switchSubMode(m);
    }

    // Chỉ cập nhật hành vi NỘP DRES. Không đổi currentAppMode hoặc tab search.
    if (dresShortcutDesc) {
        if (m === "kis") {
            dresShortcutDesc.innerText = "Ctrl + Click vào ảnh/timeline để Nộp 1-Click KIS";
        } else if (m === "qa") {
            dresShortcutDesc.innerText = "Ctrl + Click vào ảnh/timeline để Nhập & Nộp Q&A";
        } else if (m === "trake") {
            dresShortcutDesc.innerText = "Ctrl + Click vào ảnh để Mở Timeline & Chọn Chuỗi TRAKE";
        }
    }

    // Cập nhật thanh chọn chuỗi nộp TRAKE nếu Timeline đang mở.
    const trakeBar = document.getElementById("trake-timeline-bar");
    if (trakeBar) {
        trakeBar.style.display = (m === "trake") ? "flex" : "none";
    }

    // Video Player: chỉ submission TRAKE có nút nộp chuỗi riêng.
    if (videoTrakeSubmitBtn) {
        videoTrakeSubmitBtn.style.display = (m === "trake") ? "inline-flex" : "none";
    }
    if (videoModeHint) {
        const modeHints = {
            kis: { color: "#10b981", text: "KIS — Nộp ngay" },
            qa: { color: "#38bdf8", text: "QA — Mở form đáp án" },
            trake: { color: "#a855f7", text: "TRAKE — Thêm/bỏ sự kiện" }
        };
        const hint = modeHints[m] || modeHints.kis;
        videoModeHint.style.color = hint.color;
        videoModeHint.innerText = hint.text;
    }
    updateTrakeTimelineUi();

    showToast(`🎯 Chế độ nộp DRES: ${m.toUpperCase()}`, "info", 1500);
}
window.setDresSubmissionMode = setDresSubmissionMode;

// Chuyển đổi Mode Nộp DRES (KIS / QA / TRAKE)
if (dresModePills && dresModePills.length > 0) {
    dresModePills.forEach(pill => {
        pill.addEventListener("click", () => {
            const mode = pill.dataset.mode;
            setDresSubmissionMode(mode);
        });
    });
}

async function fetchDresStatus() {
    try {
        const res = await fetch(DRES_STATUS_URL);
        if (!res.ok) return;
        const json = await res.json();
        if (json.status === "success" && json.data) {
            const d = json.data;
            dresState.isLoggedIn = d.is_logged_in;
            dresState.isConnected = d.is_connected;
            dresState.username = d.username || dresState.username;
            dresState.evaluationId = d.active_evaluation_id || "";
            dresState.evaluationName = d.evaluation_name || "";
            dresState.wrongCount = d.wrong_count || 0;

            if (dresSubCount) dresSubCount.innerText = d.total_submitted || 0;

            // Cập nhật giao diện Status
            if (d.is_connected) {
                if (dresDot) dresDot.className = "dres-dot pulse-green";
                if (dresStatusLabel) dresStatusLabel.innerText = `DRES: ${d.username} (Online)`;
                if (dresEvalName) {
                    const shortRun = d.active_evaluation_id ? ` · ${d.active_evaluation_id.slice(0, 8)}…` : "";
                    dresEvalName.innerText = d.evaluation_name ? `${d.evaluation_name}${shortRun}` : "Đang chờ Task...";
                    dresEvalName.title = d.active_evaluation_id ? `Run ID: ${d.active_evaluation_id}` : "";
                }
            } else if (d.is_logged_in) {
                if (dresDot) dresDot.className = "dres-dot pulse-green";
                if (dresStatusLabel) dresStatusLabel.innerText = `DRES: ${d.username} (Chờ Eval)`;
                if (dresEvalName) dresEvalName.innerText = "Chưa có Eval ACTIVE";
            } else {
                if (dresDot) dresDot.className = "dres-dot pulse-red";
                if (dresStatusLabel) dresStatusLabel.innerText = "DRES: Chưa Đăng Nhập";
                if (dresEvalName) dresEvalName.innerText = "Offline";
            }

            // Đồng bộ lịch sử nộp bài vào modal
            if (d.history && dresHistoryList) {
                renderDresHistory(d.history);
            }

            const savedRunId = localStorage.getItem(DRES_RUN_ID_STORAGE_KEY) || "";
            if (d.is_logged_in && savedRunId && savedRunId !== d.active_evaluation_id && !applyingSavedRunId) {
                applyingSavedRunId = true;
                try {
                    const applied = await setDresRunId(savedRunId, { persist: false, quiet: true });
                    if (applied.status !== "success") {
                        showToast(`Không thể khôi phục Run ID đã lưu: ${applied.message}`, "warning", 5000);
                    }
                } finally {
                    applyingSavedRunId = false;
                }
            }
        }
    } catch (e) {
        if (dresDot) dresDot.className = "dres-dot pulse-red";
        if (dresStatusLabel) dresStatusLabel.innerText = "DRES: Mất kết nối Server Local";
    }
}

function renderDresHistory(historyItems) {
    if (!dresHistoryList) return;
    if (!historyItems || historyItems.length === 0) {
        dresHistoryList.innerHTML = `<div style="text-align: center; color: #94a3b8; padding: 20px;">Chưa có lần nộp bài nào trong câu này.</div>`;
        return;
    }

    dresHistoryList.innerHTML = historyItems.map(item => {
        const isCor = item.result === "CORRECT";
        const isWr = item.result === "WRONG";
        const cls = isCor ? "res-correct" : (isWr ? "res-wrong" : "");
        const icon = isCor ? "🎉" : (isWr ? "❌" : "📤");
        const detailStr = item.task_type === "KIS"
            ? `${item.video_id} (${item.start_ms}ms)`
            : (item.task_type === "QA" ? item.qa_text : item.trake_text);

        return `
            <div class="dres-history-item ${cls}">
                <div>
                    <span style="font-weight: 700; color: #38bdf8;">[${item.timestamp}] ${item.task_type}:</span>
                    <span style="color: #f8fafc; margin-left: 6px;">${detailStr}</span>
                </div>
                <div style="font-weight: 700;">
                    ${icon} ${item.result}
                </div>
            </div>
        `;
    }).join("");
}

// BỘ ĐIỀU PHỐI NỘP BÀI SIÊU TỐC (CTRL + CLICK HOẶC PHÍM TẮT)
// Khi DRES offline, vẫn chạy trọn thao tác cục bộ để người dùng luyện/test UI.
// Dry-run không ghi history, dedup hay trạng thái "đã nộp".
function runDresOfflineDryRun(taskType, detail, targetEl = null) {
    const normalizedTask = String(taskType || "KIS").toUpperCase();

    if (targetEl?.tagName === "BUTTON") {
        const originalText = targetEl.innerText;
        targetEl.disabled = false;
        targetEl.classList.remove("submitting");
        targetEl.classList.add("dres-dry-run");
        targetEl.innerText = `🧪 ${normalizedTask} DRY-RUN`;
        window.setTimeout(() => {
            targetEl.classList.remove("dres-dry-run");
            if (!targetEl.classList.contains("submitted")) targetEl.innerText = originalText;
        }, 1600);
    } else if (targetEl?.classList) {
        const dryRunClass = targetEl.classList.contains("timeline-item")
            ? "timeline-dres-dry-run"
            : "card-dres-dry-run";
        targetEl.classList.add(dryRunClass);
        window.setTimeout(() => targetEl.classList.remove(dryRunClass), 1600);
    }

    const suffix = detail ? ` (${detail})` : "";
    showToast(
        `🧪 Đã mô phỏng thao tác nộp ${normalizedTask}${suffix}. Cần đăng nhập DRES để gửi thật.`,
        "warning",
        3200
    );
}

function openQaSubmissionModal(videoId, frameId, targetEl = null) {
    pendingQaTarget = { videoId, frameId, targetEl };
    if (qaModalTarget) qaModalTarget.innerText = `${videoId} / Frame: ${frameId}`;
    if (dresQaAnswerInput) {
        const existingAns = (typeof subQaAnswerInput !== "undefined" && subQaAnswerInput)
            ? subQaAnswerInput.value.trim()
            : "";
        dresQaAnswerInput.value = existingAns;
        updateQaPreview();
    }
    if (dresQaModal) dresQaModal.style.display = "flex";
    setTimeout(() => dresQaAnswerInput?.focus(), 60);
}

function handleDresFastSubmit(videoId, frameId, targetEl = null) {
    if (!videoId) {
        showToast("⚠️ Không tìm thấy Video ID hợp lệ để nộp!", "warning");
        return;
    }

    const currentMode = dresTaskModeSelect ? dresTaskModeSelect.value : "kis";

    if (currentMode === "qa") {
        openQaSubmissionModal(videoId, frameId, targetEl);
        return;
    }

    if (currentMode === "trake") {
        showToast(`⏱️ Đang mở Timeline video ${videoId} để chọn chuỗi TRAKE...`, "info", 1800);
        openTimeline(videoId, frameId, true);
        return;
    }

    // Chế độ KIS mặc định: 1-Click Submit
    triggerDresSubmit(videoId, frameId, targetEl);
}

async function triggerDresSubmit(videoId, frameId, targetEl = null) {
    if (!videoId) {
        showToast("⚠️ Không tìm thấy Video ID hợp lệ để nộp!", "warning");
        return;
    }
    if (!dresState.isLoggedIn) {
        runDresOfflineDryRun("KIS", `${videoId} / KF ${frameId}`, targetEl);
        return;
    }

    if (targetEl) {
        if (targetEl.tagName === "BUTTON") {
            targetEl.disabled = true;
            targetEl.innerText = "⏳ Đang nộp...";
            targetEl.classList.add("submitting");
        } else if (targetEl.classList && targetEl.classList.contains("result-card")) {
            targetEl.classList.add("card-dres-submitting");
        }
    }

    try {
        const res = await fetch(DRES_SUBMIT_KIS_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                video_id: videoId,
                frame_id: frameId,
                force: false
            })
        });

        const data = await parseDresApiResponse(res);
        handleSubmissionResponse(data, videoId, frameId, targetEl);
    } catch (e) {
        showToast(`Lỗi kỹ thuật khi nộp bài: ${e.message}`, "warning", 6000);
        if (targetEl) {
            if (targetEl.tagName === "BUTTON") {
                targetEl.disabled = false;
                targetEl.innerText = "🚀 NỘP DRES NGAY";
                targetEl.classList.remove("submitting");
            } else if (targetEl.classList) {
                targetEl.classList.remove("card-dres-submitting");
            }
        }
    }
}

async function triggerDresTrakeSubmit(videoId, frameIds, buttonEl = null) {
    if (!videoId || !frameIds || frameIds.length < 2) {
        showToast("⚠️ Cần chọn ít nhất 2 frame theo thứ tự thời gian để nộp TRAKE!", "warning");
        return;
    }
    if (!dresState.isLoggedIn) {
        runDresOfflineDryRun("TRAKE", `${videoId} / ${frameIds.length} sự kiện`, buttonEl);
        return;
    }

    if (buttonEl) {
        buttonEl.disabled = true;
        buttonEl.innerText = "⏳ Đang nộp TRAKE...";
    }

    try {
        const res = await fetch(DRES_SUBMIT_TRAKE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                video_id: videoId,
                frame_ids: frameIds,
                force: false
            })
        });

        const data = await parseDresApiResponse(res);
        handleSubmissionResponse(data, videoId, frameIds.join(","), buttonEl);
    } catch (e) {
        showToast(`Lỗi kỹ thuật khi nộp TRAKE: ${e.message}`, "warning", 6000);
        if (buttonEl) {
            buttonEl.disabled = false;
            buttonEl.innerText = "🚀 NỘP CHUỖI TRAKE";
        }
    }
}

function updateQaPreview() {
    if (!dresQaPreviewText || !pendingQaTarget) return;
    const ans = (dresQaAnswerInput ? dresQaAnswerInput.value.trim() : "") || "<ANSWER>";
    dresQaPreviewText.innerText = `QA-${ans}-${pendingQaTarget.videoId}-<time_ms>`;
}

if (dresQaAnswerInput) {
    dresQaAnswerInput.addEventListener("input", updateQaPreview);
    dresQaAnswerInput.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            e.preventDefault();
            if (dresQaModal) dresQaModal.style.display = "none";
        } else if (e.key === "Enter") {
            e.preventDefault();
            triggerDresQaSubmit();
        }
    });
}

if (dresQaSubmitBtn) {
    dresQaSubmitBtn.addEventListener("click", triggerDresQaSubmit);
}

if (dresQaModal) {
    dresQaModal.addEventListener("click", (e) => {
        if (e.target === dresQaModal) {
            dresQaModal.style.display = "none";
        }
    });
}

if (dresQaClose && dresQaModal) {
    dresQaClose.addEventListener("click", () => {
        dresQaModal.style.display = "none";
    });
}

async function triggerDresQaSubmit() {
    if (!pendingQaTarget) return;
    const ans = dresQaAnswerInput ? dresQaAnswerInput.value.trim() : "";
    if (!ans) {
        showToast("⚠️ Vui lòng nhập câu trả lời QA!", "warning");
        return;
    }
    if (!dresState.isLoggedIn) {
        const qaTarget = pendingQaTarget;
        runDresOfflineDryRun("QA", `${qaTarget.videoId} / KF ${qaTarget.frameId}`, qaTarget.targetEl);
        if (dresQaModal) dresQaModal.style.display = "none";
        return;
    }

    if (dresQaSubmitBtn) {
        dresQaSubmitBtn.disabled = true;
        dresQaSubmitBtn.innerText = "⏳ Đang nộp...";
    }

    try {
        const res = await fetch(DRES_SUBMIT_QA_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                video_id: pendingQaTarget.videoId,
                frame_id: pendingQaTarget.frameId,
                answer: ans,
                force: false
            })
        });

        const data = await parseDresApiResponse(res);
        if (dresQaModal) dresQaModal.style.display = "none";
        handleSubmissionResponse(data, pendingQaTarget.videoId, pendingQaTarget.frameId, pendingQaTarget.targetEl);
    } catch (e) {
        showToast(`Lỗi kỹ thuật khi gửi QA: ${e.message}`, "warning", 6000);
    } finally {
        if (dresQaSubmitBtn) {
            dresQaSubmitBtn.disabled = false;
            dresQaSubmitBtn.innerText = "🚀 NỘP Q&A NGAY";
        }
    }
}

function handleSubmissionResponse(data, videoId, frameId, targetEl = null) {
    const key = `${videoId}_${frameId}`;

    // Xóa trạng thái submitting trên toàn bộ cards
    document.querySelectorAll(`.result-card[data-video-id="${videoId}"]`).forEach(card => {
        card.classList.remove("card-dres-submitting");
    });

    if (data.status === "success") {
        showToast(data.message, "success", 6000);
        dresState.submittedKeys.add(key);

        // Đánh dấu visual ĐÃ NỘP DRES trên các card phù hợp
        document.querySelectorAll(`.result-card[data-video-id="${videoId}"]`).forEach(card => {
            if (!card.dataset.frameId || card.dataset.frameId === String(frameId)) {
                card.classList.add("card-dres-submitted");
                const mediaBox = card.querySelector(".card-thumb-wrapper");
                if (mediaBox && !mediaBox.querySelector(".card-submitted-badge")) {
                    const badge = document.createElement("span");
                    badge.className = "card-submitted-badge";
                    badge.innerText = "✓ ĐÃ NỘP DRES";
                    mediaBox.appendChild(badge);
                }
            }
        });

        if (targetEl && targetEl.tagName === "BUTTON") {
            targetEl.innerText = "🎉 ĐÚNG! ĐÃ GHI ĐIỂM";
            targetEl.classList.remove("submitting");
            targetEl.classList.add("submitted");
        } else if (targetEl && targetEl.classList && targetEl.classList.contains("timeline-item")) {
            targetEl.style.outline = "2px solid #10b981";
            targetEl.style.boxShadow = "0 0 14px #10b981";
        }
    } else if (data.status === "error") {
        showToast(data.message, "error", 6000);
        dresState.wrongCount = data.wrong_count || (dresState.wrongCount + 1);

        if (targetEl && targetEl.tagName === "BUTTON") {
            targetEl.disabled = false;
            targetEl.innerText = "❌ Nộp Sai (-10đ)";
            targetEl.classList.remove("submitting");
            targetEl.classList.add("submitted");
        } else if (targetEl && targetEl.classList && targetEl.classList.contains("timeline-item")) {
            targetEl.style.outline = "2px solid #ef4444";
            targetEl.style.boxShadow = "0 0 10px #ef4444";
        }
    } else if (data.status === "warning") {
        showToast(data.message, "warning", 4500);
        if (targetEl && targetEl.tagName === "BUTTON") {
            targetEl.disabled = false;
            targetEl.innerText = "⚠️ CẢNH BÁO DRES";
            targetEl.classList.remove("submitting");
        } else if (targetEl && targetEl.classList && targetEl.classList.contains("timeline-item")) {
            targetEl.style.outline = "none";
            targetEl.style.boxShadow = "none";
        }
    } else {
        showToast(data.message, "info", 4500);
        dresState.submittedKeys.add(key);
        if (targetEl && targetEl.tagName === "BUTTON") {
            targetEl.innerText = "✓ ĐÃ GỬI BÀI";
            targetEl.classList.remove("submitting");
            targetEl.classList.add("submitted");
        } else if (targetEl && targetEl.classList && targetEl.classList.contains("timeline-item")) {
            targetEl.style.outline = "2px solid #38bdf8";
        }
    }

    fetchDresStatus();
}

function resetDresTask() {
    dresState.submittedKeys.clear();
    dresState.wrongCount = 0;
    fetch(DRES_START_TASK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_duration_s: 300 })
    }).then(() => {
        showToast("🔄 Đã làm mới câu hỏi DRES!", "info", 2500);
        document.querySelectorAll(".card-dres-submitted").forEach(c => c.classList.remove("card-dres-submitted"));
        document.querySelectorAll(".card-submitted-badge").forEach(b => b.remove());
        fetchDresStatus();
    }).catch(e => console.warn(e));
}
if (dresRefreshEvalBtn) {
    dresRefreshEvalBtn.addEventListener("click", async () => {
        showToast("🔄 Đang làm mới danh sách đợt thi DRES...", "info", 1500);
        try {
            const res = await fetch(DRES_REFRESH_URL, { method: "POST" });
            const d = await res.json();
            showToast(d.message, d.status === "success" ? "success" : "info", 3000);
            fetchDresStatus();
        } catch (e) {
            showToast(`Lỗi: ${e.message}`, "error", 2000);
        }
    });
}

// Modal Cấu hình DRES
if (dresSettingsBtn && dresSettingsModal) {
    dresSettingsBtn.addEventListener("click", () => {
        dresSettingsModal.style.display = "flex";
        if (dresInputUser) dresInputUser.value = dresState.username || "team_297";
        if (dresInputServer) dresInputServer.value = dresState.serverUrl || "https://eventretrieval.one";
        if (dresInputEvalId) dresInputEvalId.value = localStorage.getItem(DRES_RUN_ID_STORAGE_KEY) || dresState.evaluationId || "";
        if (dresInputPass) dresInputPass.value = "";
    });
}
if (dresSettingsClose && dresSettingsModal) {
    dresSettingsClose.addEventListener("click", () => {
        dresSettingsModal.style.display = "none";
    });
}
if (dresBtnSaveLogin) {
    dresBtnSaveLogin.addEventListener("click", async () => {
        const user = dresInputUser ? dresInputUser.value.trim() : "";
        const pass = dresInputPass ? dresInputPass.value.trim() : "";
        const srv = dresInputServer ? dresInputServer.value.trim() : "";
        const evalId = dresInputEvalId ? dresInputEvalId.value.trim() : "";

        if (!user) {
            showToast("Vui lòng nhập Username DRES.", "warning");
            return;
        }
        if (evalId && !isUuidLike(evalId)) {
            showToast("Run ID không đúng định dạng UUID.", "warning");
            return;
        }

        dresBtnSaveLogin.disabled = true;
        dresBtnSaveLogin.innerText = "⏳ Đang kết nối...";

        try {
            const res = await fetch(DRES_LOGIN_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: user, password: pass || null, server_url: srv })
            });
            const d = await res.json();
            if (d.status === "success") {
                showToast(d.message, "success", 4000);
                localStorage.setItem("dres_username", user);
                localStorage.setItem("dres_server_url", srv);
                dresState.username = user;
                dresState.serverUrl = srv;
                if (evalId) {
                    const evalResult = await setDresRunId(evalId, { persist: true, quiet: true });
                    if (evalResult.status !== "success") {
                        showToast(evalResult.message, "warning", 5000);
                        return;
                    }
                }
                if (dresSettingsModal) dresSettingsModal.style.display = "none";
                fetchDresStatus();
            } else {
                showToast(`❌ ${d.message}`, "error", 5000);
            }
        } catch (e) {
            showToast(`Lỗi: ${e.message}`, "error", 4000);
        } finally {
            dresBtnSaveLogin.disabled = false;
            dresBtnSaveLogin.innerText = "💾 LƯU & ĐĂNG NHẬP NGAY";
        }
    });
}

if (dresCopyRunIdBtn) {
    dresCopyRunIdBtn.addEventListener("click", async () => {
        const runId = dresInputEvalId?.value.trim() || dresState.evaluationId;
        if (!runId) {
            showToast("Chưa có Run ID để copy.", "warning", 1800);
            return;
        }
        await navigator.clipboard.writeText(runId);
        showToast("Đã copy Run ID.", "success", 1500);
    });
}

// Modal Lịch sử nộp bài
if (dresHistoryToggleBtn && dresHistoryModal) {
    dresHistoryToggleBtn.addEventListener("click", () => {
        dresHistoryModal.style.display = "flex";
        fetchDresStatus();
    });
}
if (dresHistoryClose && dresHistoryModal) {
    dresHistoryClose.addEventListener("click", () => {
        dresHistoryModal.style.display = "none";
    });
}

// Khởi tạo DRES polling
setSearchControlsReady(false);
fetchSystemReadiness();
readinessPollTimer = setInterval(fetchSystemReadiness, 750);

setTimeout(() => {
    fetchDresStatus();
    setInterval(fetchDresStatus, 5000);
}, 600);
