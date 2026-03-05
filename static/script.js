document.addEventListener('DOMContentLoaded', () => {
    // ─── API endpoints ───
    const API = {
        STATUS: '/api/status',
        STATS: '/api/data/stats',
        GENERATE: '/api/generate',
        EVALUATE: '/api/evaluate',
        ANALYZE: '/api/analyze'
    };

    // ─── Tab Navigation ───
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            // Add active to targeted
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');

            // Lazy load evaluation tab
            if (targetId === 'tab-evaluate' && !window.evalLoaded) {
                loadEvaluation();
            }
        });
    });

    // ─── Range Sliders UI ───
    document.getElementById('num-molecules').addEventListener('input', e => {
        document.getElementById('num-molecules-val').textContent = e.target.value;
    });
    document.getElementById('temperature').addEventListener('input', e => {
        document.getElementById('temperature-val').textContent = parseFloat(e.target.value).toFixed(1);
    });

    // ─── Initial Load (Status & Stats) ───
    fetchStatus();
    fetchStats();

    function fetchStatus() {
        fetch(API.STATUS)
            .then(r => r.json())
            .then(data => {
                // Update device
                document.getElementById('device-info').innerHTML =
                    `<i class="fa-solid fa-microchip"></i> ${data.device.toUpperCase()}`;

                // Update states
                updateStatusIndicator('status-dataset', data.dataset_loaded,
                    `<i class="fa-solid fa-box-archive"></i> ${data.dataset_size.toLocaleString()} Mols`);
                updateStatusIndicator('status-vae', data.vae_ready,
                    `<i class="fa-solid fa-brain"></i> VAE Ready`);
                updateStatusIndicator('status-gnn', data.gnn_ready,
                    `<i class="fa-solid fa-network-wired"></i> GNN Ready`);
            })
            .catch(err => console.error("Status error:", err));
    }

    function updateStatusIndicator(id, isReady, okText) {
        const el = document.getElementById(id);
        if (isReady) {
            el.className = 'status-indicator ready';
            el.innerHTML = okText;
        } else {
            el.className = 'status-indicator error';
            el.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Not Ready`;
        }
    }

    function fetchStats() {
        fetch(API.STATS)
            .then(r => r.json())
            .then(data => {
                if (data.error) return;
                const container = document.getElementById('overview-metrics');
                container.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-label">Total Molecules</div>
                        <div class="metric-value">${data.total.toLocaleString()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Mol Weight</div>
                        <div class="metric-value">${data.avg_mol_wt} <span style="font-size:0.5em;color:#94a3b8">Da</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg QED</div>
                        <div class="metric-value">${data.avg_qed}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg LogP</div>
                        <div class="metric-value">${data.avg_logp}</div>
                    </div>
                `;
            });
    }

    // ─── Generation Logic ───
    const generateBtn = document.getElementById('btn-generate');

    generateBtn.addEventListener('click', () => {
        const n = document.getElementById('num-molecules').value;
        const temp = document.getElementById('temperature').value;
        const mode = document.getElementById('sampling-mode').value;

        generateBtn.disabled = true;
        generateBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Generating...`;

        document.getElementById('generation-results').classList.remove('hidden');
        document.getElementById('molecules-grid').innerHTML =
            `<div style="grid-column: 1/-1; text-align:center; padding: 2rem;"><i class="fa-solid fa-atom fa-spin fa-2x"></i><br><br>Sampling from latent space...</div>`;

        fetch(API.GENERATE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n: n, temperature: temp, mode: mode })
        })
            .then(r => r.json())
            .then(data => {
                renderGeneratedMolecules(data);
            })
            .catch(err => {
                document.getElementById('molecules-grid').innerHTML = `<div style="color:red">Error: ${err}</div>`;
            })
            .finally(() => {
                generateBtn.disabled = false;
                generateBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Generate Molecules`;
            });
    });

    function renderGeneratedMolecules(data) {
        if (data.error) {
            document.getElementById('molecules-grid').innerHTML = `<div style="color:red; grid-column:1/-1">${data.error}</div>`;
            return;
        }

        // Update validity badge
        const badge = document.getElementById('validity-badge');
        const pct = (data.validity_rate * 100).toFixed(0);
        badge.innerHTML = `<i class="fa-solid fa-check-circle"></i> Validity: ${pct}% (${data.valid_count}/${data.total_generated})`;
        if (pct > 50) badge.style.color = '#81c784'; else badge.style.color = '#e57373';

        const grid = document.getElementById('molecules-grid');
        grid.innerHTML = '';

        data.molecules.forEach(mol => {
            if (mol.is_valid) {
                let gnnHTML = '';
                if (mol.gnn_predictions) {
                    gnnHTML = `<div class="mol-gnn">🔬 GNN: MW=${mol.gnn_predictions.MolWt} • LogP=${mol.gnn_predictions.LogP}</div>`;
                }

                const lipClass = mol.lipinski_pass ? 'lipinski-pass' : 'lipinski-fail';
                const lipIcon = mol.lipinski_pass ? 'fa-check' : 'fa-xmark';

                grid.innerHTML += `
                    <div class="mol-card">
                        <div class="mol-lipinski ${lipClass}"><i class="fa-solid ${lipIcon}"></i> Lipinski</div>
                        <div class="mol-img-container">
                            <img src="data:image/png;base64,${mol.image_b64}" alt="Molecule Image">
                        </div>
                        <div class="mol-smiles" title="${mol.smiles}">${mol.smiles}</div>
                        <div class="mol-props">
                            <span>MW: ${mol.properties.MolWt}</span>
                            <span>LogP: ${mol.properties.LogP}</span>
                            <span>QED: ${mol.properties.QED}</span>
                        </div>
                        ${gnnHTML}
                    </div>
                `;
            } else {
                grid.innerHTML += `
                    <div class="mol-card mol-invalid">
                        <div class="mol-img-container">
                            <div><i class="fa-solid fa-triangle-exclamation"></i> Invalid SMILES</div>
                        </div>
                        <div class="mol-smiles">${mol.smiles}</div>
                    </div>
                `;
            }
        });
    }

    // ─── Explore Tab Logic ───
    const exploreBtn = document.querySelector('[data-target="tab-explore"]');
    const applyFiltersBtn = document.getElementById('btn-apply-filters');
    const compareBtn = document.getElementById('btn-compare');
    let exploreLoaded = false;

    exploreBtn.addEventListener('click', () => {
        if (!exploreLoaded) {
            exploreLoaded = true;
            fetchAndRenderExplore();
        }
    });

    applyFiltersBtn.addEventListener('click', () => {
        applyFiltersBtn.disabled = true;
        applyFiltersBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Applying...`;
        fetchAndRenderExplore();
    });

    function fetchAndRenderExplore() {
        // Read filters
        const mwMin = document.getElementById('filter-mw-min').value;
        const mwMax = document.getElementById('filter-mw-max').value;
        const logpMin = document.getElementById('filter-logp-min').value;
        const logpMax = document.getElementById('filter-logp-max').value;
        const qedMin = document.getElementById('filter-qed-min').value;
        const qedMax = document.getElementById('filter-qed-max').value;

        const query = `?molwt_min=${mwMin}&molwt_max=${mwMax}&logp_min=${logpMin}&logp_max=${logpMax}&qed_min=${qedMin}&qed_max=${qedMax}&limit=50`;

        fetch('/api/explore' + query)
            .then(r => r.json())
            .then(data => {
                applyFiltersBtn.disabled = false;
                applyFiltersBtn.innerHTML = `<i class="fa-solid fa-filter"></i> Apply Filters`;

                if (data.error) {
                    document.getElementById('explore-stats').innerHTML = `<span style="color:red"><i class="fa-solid fa-triangle-exclamation"></i> ${data.error} (Generate molecules first)</span>`;
                    document.getElementById('explore-table-body').innerHTML = `<tr><td colspan="6" style="text-align:center">No data available</td></tr>`;
                    return;
                }

                document.getElementById('explore-stats').innerHTML = `Showing <b>${data.total_filtered}</b> of ${data.total_available} generated molecules`;

                const tbody = document.getElementById('explore-table-body');
                tbody.innerHTML = '';

                data.molecules.forEach(mol => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><input type="checkbox" class="compare-cb" value="${mol.SMILES}"></td>
                        <td class="smiles-col" title="${mol.SMILES}">${mol.SMILES}</td>
                        <td>${parseFloat(mol.MolWt).toFixed(1)}</td>
                        <td>${parseFloat(mol.LogP).toFixed(2)}</td>
                        <td>${mol.QED ? parseFloat(mol.QED).toFixed(3) : 'N/A'}</td>
                        <td>${mol.TPSA ? parseFloat(mol.TPSA).toFixed(1) : 'N/A'}</td>
                    `;
                    tbody.appendChild(tr);
                });

                // Attach checkbox listeners
                document.querySelectorAll('.compare-cb').forEach(cb => {
                    cb.addEventListener('change', updateCompareBtn);
                });
                updateCompareBtn();
            });
    }

    function updateCompareBtn() {
        if (!compareBtn) return;
        const checked = document.querySelectorAll('.compare-cb:checked');
        if (checked.length > 0 && checked.length <= 4) {
            compareBtn.disabled = false;
            compareBtn.innerHTML = `<i class="fa-solid fa-vial"></i> Compare ${checked.length} Selected`;
        } else {
            compareBtn.disabled = true;
            if (checked.length > 4) {
                compareBtn.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Max 4 allowed`;
            } else {
                compareBtn.innerHTML = `<i class="fa-solid fa-vial"></i> Compare`;
            }
        }
    }

    if (compareBtn) {
        compareBtn.addEventListener('click', async () => {
            const checked = document.querySelectorAll('.compare-cb:checked');
            if (checked.length === 0 || checked.length > 4) return;

            compareBtn.disabled = true;
            compareBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...`;

            const grid = document.getElementById('compare-grid');
            grid.classList.remove('hidden');
            grid.innerHTML = '';

            for (let cb of checked) {
                const smiles = cb.value;
                try {
                    const res = await fetch(API.ANALYZE, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ smiles: smiles })
                    });
                    const data = await res.json();

                    if (data.is_valid) {
                        const lipClass = data.lipinski_pass ? 'lipinski-pass' : 'lipinski-fail';
                        const lipIcon = data.lipinski_pass ? 'fa-check' : 'fa-xmark';
                        grid.innerHTML += `
                            <div class="mol-card">
                                <div class="mol-lipinski ${lipClass}"><i class="fa-solid ${lipIcon}"></i></div>
                                <div class="mol-img-container">
                                    <img src="data:image/png;base64,${data.image_b64}">
                                </div>
                                <div class="mol-smiles" title="${smiles}">${smiles}</div>
                                <div class="mol-props">
                                    <span>MW: ${data.properties.MolWt}</span>
                                    <span>LogP: ${data.properties.LogP}</span>
                                </div>
                                <div class="mol-props" style="border:none; padding-top:0">
                                    <span>QED: ${data.properties.QED}</span>
                                    <span>TPSA: ${data.properties.TPSA}</span>
                                </div>
                            </div>
                        `;
                    }
                } catch (e) {
                    console.error("Compare error", e);
                }
            }

            compareBtn.disabled = false;
            compareBtn.innerHTML = `<i class="fa-solid fa-vial"></i> Compare ${checked.length} Selected`;
        });
    }

    // ─── Analyzer Logic ───
    document.getElementById('btn-analyze').addEventListener('click', () => {
        const smiles = document.getElementById('custom-smiles').value.trim();
        if (!smiles) return;

        const btn = document.getElementById('btn-analyze');
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i>`;

        fetch(API.ANALYZE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles: smiles })
        })
            .then(r => r.json())
            .then(data => {
                const container = document.getElementById('analyzer-results');
                container.classList.remove('hidden');

                if (!data.is_valid) {
                    container.innerHTML = `<div style="color:red;grid-column:1/-1"><i class="fa-solid fa-circle-xmark"></i> Invalid SMILES string: ${smiles}</div>`;
                    return;
                }

                let propsHTML = '';
                for (const [k, v] of Object.entries(data.properties)) {
                    propsHTML += `<div style="display:flex; justify-content:space-between; margin-bottom:0.5rem; border-bottom:1px solid rgba(255,255,255,0.05)"><span>${k}</span> <span style="color:#00e5ff; font-weight:bold">${v}</span></div>`;
                }

                let gnnHTML = '';
                if (data.gnn_predictions) {
                    gnnHTML = `<h4 style="margin: 1.5rem 0 0.5rem 0; color:#76ff03"><i class="fa-solid fa-network-wired"></i> GNN Predictions</h4>`;
                    for (const [k, v] of Object.entries(data.gnn_predictions)) {
                        gnnHTML += `<div style="display:flex; justify-content:space-between; margin-bottom:0.5rem; border-bottom:1px solid rgba(255,255,255,0.05)"><span>${k}</span> <span style="color:#76ff03; font-weight:bold">${v}</span></div>`;
                    }
                }

                container.innerHTML = `
                <div style="background:white; padding:1rem; border-radius:8px; display:flex; align-items:center; justify-content:center;">
                    <img src="data:image/png;base64,${data.image_b64}" style="max-width:100%">
                </div>
                <div>
                    <h4 style="margin-bottom:0.5rem; color:#00bcd4"><i class="fa-solid fa-atom"></i> Exact Properties</h4>
                    ${propsHTML}
                    <div style="margin-top:1rem; padding: 0.5rem; background: ${data.lipinski_pass ? 'rgba(76,175,80,0.1)' : 'rgba(244,67,54,0.1)'}; color: ${data.lipinski_pass ? '#81c784' : '#e57373'}; border-radius:4px; text-align:center;">
                        Rule of 5: <b>${data.lipinski_pass ? 'PASS' : 'FAIL'}</b>
                    </div>
                    ${gnnHTML}
                </div>
            `;
            })
            .finally(() => {
                btn.disabled = false;
                btn.innerHTML = `<i class="fa-solid fa-magnifying-glass"></i> Analyze`;
            });
    });

    // ─── Evaluation Tab Lazy Load ───
    function loadEvaluation() {
        window.evalLoaded = true;
        fetch(API.EVALUATE)
            .then(r => r.json())
            .then(data => {
                const container = document.getElementById('eval-metrics-container');
                if (data.error) {
                    container.innerHTML = `<div style="color:#94a3b8"><i class="fa-solid fa-circle-info"></i> Pipeline not run yet. Generate dataset evaluation via backend script: <code>python src/evaluate.py</code></div>`;
                    return;
                }

                if (data.status === 'partial') {
                    // Just show training stats
                    container.innerHTML = `<div style="color:#00e5ff; margin-bottom:1rem">Training Metrics Found (Full evaluation pending)</div>`;
                    // render basic view...
                    return;
                }

                const m = data.metrics;
                container.innerHTML = `
                    <div class="metrics-grid" style="grid-template-columns: repeat(4, 1fr)">
                        <div class="metric-card"><div class="metric-label">Validation Rate</div><div class="metric-value">${(m.validity * 100).toFixed(1)}%</div></div>
                        <div class="metric-card"><div class="metric-label">Uniqueness</div><div class="metric-value">${(m.uniqueness * 100).toFixed(1)}%</div></div>
                        <div class="metric-card"><div class="metric-label">Novelty</div><div class="metric-value">${(m.novelty * 100).toFixed(1)}%</div></div>
                        <div class="metric-card"><div class="metric-label">Lipinski Pass</div><div class="metric-value">${(m.lipinski_rate * 100).toFixed(1)}%</div></div>
                    </div>
                    
                    <h4 style="margin:2rem 0 1rem; color:#00e5ff"><i class="fa-solid fa-image"></i> Evaluation Plots</h4>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
                        <img src="/api/assets/evaluation_summary.png" style="width:100%; border-radius:8px; border:1px solid rgba(0,188,212,0.2)" onerror="this.style.display='none'">
                        <img src="/api/assets/generated_molecules.png" style="width:100%; border-radius:8px; border:1px solid rgba(0,188,212,0.2)" onerror="this.style.display='none'">
                    </div>
                `;
            });
    }
});
