// Add some interactive elements
        document.addEventListener('DOMContentLoaded', function() {
            // Animate confidence bar
            const confidenceFill = document.querySelector('.confidence-fill');
            if (confidenceFill) {
                const width = confidenceFill.style.width;
                confidenceFill.style.width = '0%';
                setTimeout(() => {
                    confidenceFill.style.width = width;
                }, 500);
            }
            
            // Add keyboard shortcut for new analysis
            document.addEventListener('keydown', function(e) {
                if (e.key === 'n' && e.ctrlKey) {
                    e.preventDefault();
                    window.location.href = '/';
                }
            });
        });
        
        // Print-friendly styles
        const printStyles = `
            @media print {
                body { background: white !important; color: black !important; }
                .result-container, .tips-section { 
                    background: white !important; 
                    border: 1px solid #ccc !important; 
                }
                .prediction-badge { color: black !important; }
                .action-buttons { display: none; }
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.textContent = printStyles;
        document.head.appendChild(styleSheet);