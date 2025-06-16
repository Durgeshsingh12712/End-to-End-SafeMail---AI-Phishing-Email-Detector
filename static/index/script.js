// Create animated background particles
        function createParticles() {
            const particles = document.getElementById('particles');
            const particleCount = 50;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 6 + 's';
                particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
                particles.appendChild(particle);
            }
        }

        // Initialize particles
        createParticles();

        // Form handling
        document.getElementById('emailForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const emailText = document.getElementById('emailText').value.trim();
            const checkButton = document.getElementById('checkButton');
            const loading = document.getElementById('loading');
            const errorMessage = document.getElementById('errorMessage');
            const resultSection = document.getElementById('resultSection');
            
            if (!emailText) {
                showError('Please enter email content to analyze.');
                return;
            }
            
            // Show loading state
            checkButton.disabled = true;
            loading.style.display = 'block';
            errorMessage.style.display = 'none';
            resultSection.style.display = 'none';
            
            try {
                const formData = new FormData();
                formData.append('email_text', emailText);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    // For successful form submission, Flask will redirect to result page
                    // For this demo, we'll use the API endpoint instead
                    const apiResponse = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: emailText })
                    });
                    
                    if (apiResponse.ok) {
                        const result = await apiResponse.json();
                        showResult(result);
                    } else {
                        const error = await apiResponse.json();
                        showError(error.error || 'Prediction failed');
                    }
                } else {
                    const error = await response.json();
                    showError(error.error || 'Request failed');
                }
            } catch (error) {
                showError('Network error. Please try again.');
                console.error('Error:', error);
            } finally {
                checkButton.disabled = false;
                loading.style.display = 'none';
            }
        });
        
        function showResult(result) {
            const resultSection = document.getElementById('resultSection');
            const predictionBadge = document.getElementById('predictionBadge');
            const confidenceText = document.getElementById('confidenceText');
            const confidenceFill = document.getElementById('confidenceFill');
            const resultExplanation = document.getElementById('resultExplanation');
            
            // Update prediction badge
            const isPhishing = result.prediction.toLowerCase() === 'phishing';
            predictionBadge.textContent = isPhishing ? '⚠️ Phishing Detected' : '✅ Safe Email';
            predictionBadge.className = `prediction-badge ${isPhishing ? 'prediction-phishing' : 'prediction-safe'}`;
            
            // Update confidence
            const confidence = Math.round(result.confidence * 100);
            confidenceText.textContent = `${confidence}%`;
            confidenceFill.style.width = `${confidence}%`;
            
            // Update explanation
            const explanation = isPhishing 
                ? `<p><strong>⚠️ Warning:</strong> This email shows characteristics commonly associated with phishing attempts. Please verify the sender's identity through official channels before taking any action.</p>
                   <p><strong>Recommended actions:</strong></p>
                   <ul>
                       <li>Do not click any links or download attachments</li>
                       <li>Verify with the sender using known contact information</li>
                       <li>Report to your IT security team if applicable</li>
                   </ul>`
                : `<p><strong>✅ Good news:</strong> This email appears to be legitimate based on our analysis. However, always remain vigilant and trust your instincts.</p>
                   <p><strong>General safety tips:</strong></p>
                   <ul>
                       <li>Still verify unexpected requests for sensitive information</li>
                       <li>Check URLs carefully before clicking</li>
                       <li>Keep your security software updated</li>
                   </ul>`;
            
            resultExplanation.innerHTML = explanation;
            
            // Show result section
            resultSection.style.display = 'block';
            resultSection.scrollIntoView({ behavior: 'smooth' });
        }
        
        function showError(message) {
            const errorMessage = document.getElementById('errorMessage');
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
        }
        
        // Keyboard navigation support
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                document.getElementById('emailForm').dispatchEvent(new Event('submit'));
            }
        });
        
        // Auto-resize textarea
        const textarea = document.getElementById('emailText');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.max(200, this.scrollHeight) + 'px';
        });