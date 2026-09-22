document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('advisor-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    
    const emptyState = document.getElementById('empty-state');
    const resultsList = document.getElementById('results-list');
    const resultsCount = document.getElementById('results-count');
    
    const detailModal = document.getElementById('detail-modal');
    const closeModal = detailModal.querySelector('.close-modal');
    const modalBodyContent = document.getElementById('modal-body-content');

    // Global variable to store active recommendations
    let currentRecommendations = [];

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Disable button & show spinner
        submitBtn.disabled = true;
        btnText.style.opacity = '0.3';
        loader.classList.remove('hidden');
        
        // Gather values
        const startedRadio = document.querySelector('input[name="started"]:checked');
        const coFinancingRadio = document.querySelector('input[name="co_financing"]:checked');
        
        // Show skeletons
        showSkeletons();
        
        const payload = {
            location: document.getElementById('location').value,
            applicant_type: document.getElementById('applicant_type').value,
            company_size: document.getElementById('company_size').value,
            started: startedRadio ? startedRadio.value === 'true' : false,
            co_financing: coFinancingRadio ? coFinancingRadio.value === 'true' : true,
            sector: document.getElementById('sector').value,
            goal: document.getElementById('goal').value,
            funding_type: document.getElementById('funding_type').value,
            description: document.getElementById('description').value
        };
        
        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                throw new Error('API request failed');
            }
            
            currentRecommendations = await response.json();
            renderResults(currentRecommendations);
            
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            alert('Fehler bei der Verbindung zum Server. Bitte stellen Sie sicher, dass das Backend läuft.');
        } finally {
            // Re-enable button & hide spinner
            submitBtn.disabled = false;
            btnText.style.opacity = '1';
            loader.classList.add('hidden');
        }
    });

    function showSkeletons() {
        resultsCount.textContent = 'Analysiere...';
        emptyState.classList.add('hidden');
        resultsList.classList.remove('hidden');
        resultsList.innerHTML = '';
        
        for(let i=0; i<3; i++) {
            resultsList.innerHTML += `
                <div class="skeleton-card">
                    <div class="skeleton-line title"></div>
                    <div class="skeleton-line full"></div>
                    <div class="skeleton-line full"></div>
                    <div class="skeleton-line medium"></div>
                    <div style="margin-top: 1.5rem">
                        <div class="skeleton-line short"></div>
                        <div class="skeleton-line short"></div>
                    </div>
                </div>
            `;
        }
    }

    function renderResults(recommendations) {
        // Clear old results
        resultsList.innerHTML = '';
        
        if (recommendations.length === 0) {
            resultsCount.textContent = '0 Programme gefunden';
            resultsList.classList.add('hidden');
            emptyState.innerHTML = `
                <div class="empty-icon">⚠️</div>
                <h3>Keine passenden Programme gefunden</h3>
                <p>Für Ihre gewählten Kriterien konnten keine Förderprogramme ermittelt werden. Bitte passen Sie Ihre Filter an.</p>
            `;
            emptyState.classList.remove('hidden');
            return;
        }
        
        resultsCount.textContent = `${recommendations.length} Programme gefunden`;
        emptyState.classList.add('hidden');
        resultsList.classList.remove('hidden');
        
        // Icons
        const checkIcon = `<svg class="icon-svg icon-success" viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        const warnIcon = `<svg class="icon-svg icon-warning" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;

        recommendations.forEach((rec, index) => {
            const card = document.createElement('div');
            card.className = 'funding-card animate-fade-in-up';
            card.style.animationDelay = `${index * 0.08}s`;
            
            // Confidence badge
            const confidenceMap = {
                high: { label: 'Hohe Übereinstimmung', cls: 'confidence-high' },
                medium: { label: 'Mögliche Übereinstimmung', cls: 'confidence-medium' },
                low: { label: 'Schwache Übereinstimmung', cls: 'confidence-low' }
            };
            const conf = confidenceMap[rec.confidence] || confidenceMap.low;
            
            // Format reasons (highlight warnings)
            const reasonsHtml = rec.reasons.map(reason => {
                const isWarning = reason.startsWith('⚠️');
                const text = reason.replace('⚠️ ', '').replace('⚠️', '');
                return `<li class="explanation-item${isWarning ? ' exclusion-warning' : ''}">${isWarning ? warnIcon : checkIcon} <span style="margin-left: 6px">${text}</span></li>`;
            }).join('');
            
            // Circular ring
            const scoreDash = `${rec.score}, 100`;
            
            card.innerHTML = `
                <div class="card-header">
                    <h3 class="card-title">${rec.title}</h3>
                    <div class="card-scores">
                        <span class="confidence-badge ${conf.cls}">${conf.label}</span>
                        <div class="card-score-ring ${conf.cls}">
                            <svg viewBox="0 0 36 36" class="circular-chart">
                                <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                <path class="circle" stroke-dasharray="${scoreDash}" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                <text x="18" y="20.35" class="percentage">${rec.score}%</text>
                            </svg>
                        </div>
                    </div>
                </div>
                <p class="card-teaser">${rec.summary || rec.teaser || 'Keine Kurzbeschreibung vorhanden.'}</p>
                
                <div class="card-explanation">
                    <div class="explanation-title">Warum empfohlen?</div>
                    <ul class="explanation-list">
                        ${reasonsHtml}
                    </ul>
                </div>
                
                <button class="detail-btn" data-index="${index}">Details anzeigen</button>
            `;
            
            resultsList.appendChild(card);
        });

        // Add event listeners to detail buttons
        const detailButtons = resultsList.querySelectorAll('.detail-btn');
        detailButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.getAttribute('data-index'));
                openDetailsModal(idx);
            });
        });
    }

    function openDetailsModal(index) {
        const rec = currentRecommendations[index];
        if (!rec) return;
        
        // Clean or parse body text/HTML
        let contentHtml = rec.body_html || `<p>${rec.summary}</p>`;
        
        // Remove CDATA tags if present in HTML
        contentHtml = contentHtml.replace('<![CDATA[', '').replace(']]>', '');
        
        // Convert xlink:href="target:/BMWI..." to standard clickable href="https://www.foerderdatenbank.de..."
        contentHtml = contentHtml.replace(/xlink:href="target:\/BMWI([^"]+)"/g, 'href="https://www.foerderdatenbank.de$1.html" target="_blank"');
        contentHtml = contentHtml.replace(/xlink:href="([^"]+)"/g, 'href="$1" target="_blank"');
        
        modalBodyContent.innerHTML = `
            <h1>${rec.title}</h1>
            <div style="margin-top: 1.5rem; color: var(--text-secondary); display: flex; gap: 1rem;">
                <span class="badge" style="background: var(--success-glow); color: var(--success-green); border-color: rgba(16, 185, 129, 0.2)">
                    Match: ${rec.score}%
                </span>
                <span class="badge">Programm-ID: ${rec.id}</span>
            </div>
            <div class="modal-body-text" style="margin-top: 2rem;">
                ${contentHtml}
            </div>
        `;
        
        detailModal.classList.remove('hidden');
    }

    // Modal close events
    closeModal.addEventListener('click', () => {
        detailModal.classList.add('hidden');
    });

    window.addEventListener('click', (e) => {
        if (e.target === detailModal) {
            detailModal.classList.add('hidden');
        }
    });
});
