(function () {
    'use strict';

    const baseStartSim = typeof startSim === 'function' ? startSim : null;
    const simulateBtn = document.getElementById('simulate-btn');
    const narrativeInput = document.getElementById('narrative');
    const fileInputEl = document.getElementById('file-input');
    const deployBtn = document.getElementById('btn-deploy');
    const deployInput = document.getElementById('media-injection');
    const overlay = document.getElementById('injection-overlay');
    const payloadLayer = document.getElementById('new-media-layer');
    const overlayStatus = document.getElementById('overlay-status');
    const analysisParas = Array.from(document.querySelectorAll('.ai-para'));
    const globeLabelIds = ['lbl1', 'lbl2', 'lbl3', 'lbl4'];
    const scrollBlockIds = ['sb1', 'sb2', 'sb3'];

    let currentJobId = null;
    let eventSource = null;
    let networkRenderTimer = null;
    let act3ShownByLiveUpdates = false;
    window.liveBackendRunActive = false;

    function formatNumber(value) {
        return Number(value || 0).toLocaleString();
    }

    function setCounterValue(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = formatNumber(value);
        }
    }

    function setText(id, value) {
        const el = document.getElementById(id);
        if (el && value != null) {
            el.textContent = value;
        }
    }

    function clearLiveConnection() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        currentJobId = null;
        window.liveBackendRunActive = false;
    }

    function resetPipelineDataForRun(narrative) {
        window.pipelineData = {
            job_id: null,
            narrative: narrative,
            status: 'queued',
            progress: { stage: 'queued', processed: 0, total: 5000, percent: 0 },
            stats: { impressions: 0, likes: 0, dislikes: 0, shares: 0, comments: 0, nothing: 0 },
            reaction_bar: { liked: 0, disliked: 0, shared: 0, comment: 0, none: 100 },
            chart: {
                labels: ['0%'],
                series: [
                    { key: 'liked', label: 'Liked', color: '#10b981', values: [0] },
                    { key: 'neutral', label: 'Neutral', color: '#9ca3af', values: [100] },
                    { key: 'disliked', label: 'Disliked', color: '#ef4444', values: [0] }
                ]
            },
            graph: { nodes: [], links: [] },
            shares: [],
            agents: [],
            analysis: {
                globe_status: 'Starting backend simulation...',
                globe_labels: [
                    'Preparing first response cluster...',
                    'Preparing first response cluster...',
                    'Preparing first response cluster...',
                    'Preparing first response cluster...'
                ],
                scroll_messages: [
                    'Switching from placeholder UI to backend data.',
                    'Loading the embedding and predictor stack.',
                    'Charts will fill in as batches complete.'
                ],
                paragraphs: [
                    'The site is now waiting on the backend simulation rather than static JSON.',
                    'As the model finishes each batch, the chart, counters, and graph update in place.',
                    'Final analysis text is generated from the run output.'
                ]
            }
        };
    }

    function applySnapshot(snapshot) {
        window.pipelineData = snapshot;

        const analysis = snapshot.analysis || {};
        const stats = snapshot.stats || {};
        const rxn = snapshot.reaction_bar || {};

        if (typeof status !== 'undefined' && analysis.globe_status) {
            status.classList.add('visible');
            status.textContent = analysis.globe_status;
        }

        (analysis.globe_labels || []).slice(0, 4).forEach((text, idx) => setText(globeLabelIds[idx], text));
        (analysis.scroll_messages || []).slice(0, 3).forEach((text, idx) => setText(scrollBlockIds[idx], text));
        analysisParas.forEach((el, idx) => {
            if (analysis.paragraphs && analysis.paragraphs[idx]) {
                el.textContent = analysis.paragraphs[idx];
            }
        });

        setCounterValue('cnt-impressions', stats.impressions);
        setCounterValue('cnt-likes', stats.likes);
        setCounterValue('cnt-dislikes', stats.dislikes);
        setCounterValue('cnt-shares', stats.shares);
        setCounterValue('cnt-comments', stats.comments);

        const rxnMap = {
            liked: 'rxn-liked',
            disliked: 'rxn-disliked',
            shared: 'rxn-shared',
            comment: 'rxn-comment',
            none: 'rxn-none'
        };
        Object.keys(rxnMap).forEach((key) => {
            const bar = document.getElementById(rxnMap[key]);
            if (bar) {
                bar.style.width = String(rxn[key] || 0) + '%';
            }
            const pct = document.getElementById('rxn-pct-' + key);
            if (pct) {
                pct.textContent = Math.round(rxn[key] || 0);
            }
        });

        if (typeof drawBeliefChart === 'function') {
            drawBeliefChart();
        }

        if ((snapshot.status === 'completed' || snapshot.status === 'error') && typeof renderNetworkGraph === 'function') {
            scheduleNetworkRender();
        }

        if ((snapshot.status === 'completed' || snapshot.status === 'error') && !act3ShownByLiveUpdates && typeof showAct3 === 'function') {
            showAct3();
            act3ShownByLiveUpdates = true;
        }

        if (snapshot.status === 'completed' || snapshot.status === 'error') {
            hideDeployOverlay();
            clearLiveConnection();
        }
    }

    function scheduleNetworkRender() {
        clearTimeout(networkRenderTimer);
        networkRenderTimer = setTimeout(() => {
            const current = document.getElementById('network-graph2');
            if (!current || !current.parentNode) return;
            const fresh = current.cloneNode(false);
            current.parentNode.replaceChild(fresh, current);
            if (typeof networkRenderer !== 'undefined') {
                networkRenderer = null;
            }
            if (typeof networkScene !== 'undefined') networkScene = null;
            if (typeof networkCamera !== 'undefined') networkCamera = null;
            if (typeof networkControls !== 'undefined') networkControls = null;
            if (typeof networkNodes !== 'undefined') networkNodes = [];
            if (typeof networkLinks !== 'undefined') networkLinks = [];
            if (typeof starGroup !== 'undefined') starGroup = null;
            if (typeof renderNetworkGraph === 'function') {
                renderNetworkGraph();
            }
        }, 180);
    }

    function hideDeployOverlay() {
        if (overlay) overlay.classList.remove('active');
        if (payloadLayer) payloadLayer.classList.remove('drop');
        if (overlayStatus) overlayStatus.classList.remove('pulse');
        if (deployBtn) {
            deployBtn.innerText = 'Deploy Payload';
            deployBtn.style.opacity = '1';
            deployBtn.style.pointerEvents = 'auto';
        }
    }

    async function startBackendSimulation(narrative, options) {
        clearLiveConnection();
        window.liveBackendRunActive = true;
        resetPipelineDataForRun(narrative);
        applySnapshot(window.pipelineData);

        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                narrative: narrative,
                seed: options && options.seed != null ? options.seed : null
            })
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || 'Failed to start simulation');
        }

        currentJobId = payload.job_id;
        eventSource = new EventSource('/api/jobs/' + currentJobId + '/events');
        eventSource.onmessage = function (event) {
            try {
                applySnapshot(JSON.parse(event.data));
            } catch (err) {
                console.error('Failed to parse backend event', err);
            }
        };
        eventSource.onerror = function () {
            if (currentJobId) {
                fetch('/api/jobs/' + currentJobId)
                    .then((res) => res.ok ? res.json() : null)
                    .then((snapshot) => {
                        if (snapshot) applySnapshot(snapshot);
                    })
                    .catch(() => {});
            }
        };
    }

    async function beginRun(narrative, options) {
        if (!narrative || !narrative.trim()) {
            narrativeInput.style.borderColor = 'rgba(192,57,43,.7)';
            narrativeInput.style.boxShadow = '0 0 0 3px rgba(192,57,43,.15)';
            setTimeout(() => {
                narrativeInput.style.borderColor = '';
                narrativeInput.style.boxShadow = '';
            }, 1200);
            return;
        }

        if (!options || !options.injection) {
            act3ShownByLiveUpdates = false;
            if (baseStartSim) {
                baseStartSim();
            }
        } else if (overlay && overlayStatus && payloadLayer) {
            overlay.classList.add('active');
            overlayStatus.classList.add('pulse');
            overlayStatus.innerText = 'ASSIMILATING PAYLOAD';
            payloadLayer.classList.add('drop');
            deployBtn.innerText = 'Injecting payload...';
            deployBtn.style.opacity = '0.7';
            deployBtn.style.pointerEvents = 'none';
        }

        await startBackendSimulation(narrative, options || {});
    }

    if (simulateBtn) {
        simulateBtn.addEventListener('click', function (event) {
            event.preventDefault();
            event.stopImmediatePropagation();
            const text = (narrativeInput.value || '').trim();
            const hasFile = fileInputEl && fileInputEl.files && fileInputEl.files.length > 0;
            if (!text && !hasFile) {
                narrativeInput.style.borderColor = 'rgba(192,57,43,.7)';
                narrativeInput.style.boxShadow = '0 0 0 3px rgba(192,57,43,.15)';
                setTimeout(() => {
                    narrativeInput.style.borderColor = '';
                    narrativeInput.style.boxShadow = '';
                }, 1200);
                return;
            }
            beginRun(text, { injection: false }).catch((err) => {
                console.error(err);
                hideDeployOverlay();
            });
        }, true);
    }

    deployPayload = function () {
        const text = (deployInput && deployInput.value ? deployInput.value : '').trim();
        if (!text) return;
        beginRun(text, { injection: true }).then(() => {
            if (deployInput) {
                deployInput.value = '';
            }
        }).catch((err) => {
            console.error(err);
            hideDeployOverlay();
        });
    };

    drawBeliefChart = function () {
        const chart = (window.pipelineData && window.pipelineData.chart) || {};
        const series = Array.isArray(chart.series) ? chart.series : [];
        const labels = Array.isArray(chart.labels) && chart.labels.length ? chart.labels : ['0%'];
        if (!document.getElementById('chart-line') || !series.length) return;

        if (chartLine) chartLine.destroy();
        if (chartBar) chartBar.destroy();
        if (chartPie) chartPie.destroy();

        Chart.defaults.color = '#666';
        Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

        chartLine = new Chart(document.getElementById('chart-line'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: series.map((item) => ({
                    label: item.label,
                    data: item.values,
                    borderColor: item.color,
                    backgroundColor: item.color + '22',
                    tension: 0.35,
                    fill: true,
                    pointRadius: 3
                }))
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true, font: { size: 11 } } } },
                scales: { y: { beginAtZero: true, max: 100 } },
                animation: { duration: 450, easing: 'easeOutQuart' }
            }
        });

        const lastValues = series.map((item) => item.values[item.values.length - 1] || 0);
        const lastLabel = labels[labels.length - 1] || 'Current';

        chartBar = new Chart(document.getElementById('chart-bar'), {
            type: 'bar',
            data: {
                labels: series.map((item) => item.label),
                datasets: [{
                    label: 'Belief Shift at ' + lastLabel,
                    data: lastValues,
                    backgroundColor: series.map((item) => item.color),
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true, max: 100 } },
                animation: { duration: 450, easing: 'easeOutQuart' }
            }
        });

        chartPie = new Chart(document.getElementById('chart-pie'), {
            type: 'pie',
            data: {
                labels: series.map((item) => item.label),
                datasets: [{
                    data: lastValues,
                    backgroundColor: series.map((item) => item.color),
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, usePointStyle: true, font: { size: 11 } } } },
                animation: { duration: 450, easing: 'easeOutQuart' }
            }
        });
    };

    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', function () {
            clearLiveConnection();
            hideDeployOverlay();
            act3ShownByLiveUpdates = false;
        }, true);
    }
})();
