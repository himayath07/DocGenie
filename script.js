// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    });
});

// Form submission handling
const contactForm = document.getElementById('contact-form');
if (contactForm) {
    contactForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get form data
        const formData = new FormData(this);
        const formValues = Object.fromEntries(formData);
        
        // You would typically send this data to a server
        console.log('Form submitted:', formValues);
        
        // Show success message
        alert('Thank you for your message! We will get back to you soon.');
        
        // Reset form
        this.reset();
    });
}

// Add animation to service cards on scroll
const cards = document.querySelectorAll('.card');
const observerOptions = {
    threshold: 0.5
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

cards.forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'all 0.5s ease-in-out';
    observer.observe(card);
});

document.addEventListener('DOMContentLoaded', function() {
    // Get all necessary elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const summaryDropZone = document.getElementById('summaryDropZone');
    const summaryFileInput = document.getElementById('summaryFileInput');
    const summaryResult = document.getElementById('summaryResult');
    const conversionDropZone = document.getElementById('conversionDropZone');
    const conversionFileInput = document.getElementById('conversionFileInput');
    const conversionResult = document.getElementById('conversionResult');
    const convertFormat = document.getElementById('convertFormat');
    const youtubeUrl = document.getElementById('youtubeUrl');
    const summarizeVideo = document.getElementById('summarizeVideo');
    const youtubeResult = document.getElementById('youtubeResult');

    // Add click handlers for all upload buttons
    document.querySelectorAll('.upload-btn').forEach(button => {
        button.addEventListener('click', (e) => {
            // Find the closest input element to this button
            const fileInput = e.target.parentElement.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.click();
            }
        });
    });

    // Document Summarization
    if (summaryDropZone && summaryFileInput) {
        summaryDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            summaryDropZone.classList.add('drag-over');
        });

        summaryDropZone.addEventListener('dragleave', () => {
            summaryDropZone.classList.remove('drag-over');
        });

        summaryDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            summaryDropZone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleSummaryFile(files[0]);
            }
        });

        summaryFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleSummaryFile(e.target.files[0]);
            }
        });
    }

    // Document Conversion
    if (conversionDropZone && conversionFileInput) {
        conversionDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            conversionDropZone.classList.add('drag-over');
        });

        conversionDropZone.addEventListener('dragleave', () => {
            conversionDropZone.classList.remove('drag-over');
        });

        conversionDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            conversionDropZone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleConversionFile(files[0]);
            }
        });

        conversionFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleConversionFile(e.target.files[0]);
            }
        });
    }

    // YouTube Video Summarization
    if (summarizeVideo) {
        summarizeVideo.addEventListener('click', handleYouTubeUrl);
    }

    // File handling functions
    function handleSummaryFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        if (summaryResult) {
            // Clear previous results
            summaryResult.style.display = 'block';
            summaryResult.innerHTML = '<div class="loading">Processing your file...</div>';

            fetch('http://localhost:8080/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Received data:', data); // Debug log
                
                // Store the summary in a variable for download
                const summaryText = data.summary;
                
                // Create a formatted display of the summary
                summaryResult.innerHTML = `
                    <div class="result-content">
                        <h4>Document Summary</h4>
                        <div class="summary-box">
                            ${summaryText}
                        </div>
                        <div class="metadata">
                            <p>File processed: ${file.name}</p>
                            <p>Size: ${(file.size / 1024).toFixed(2)} KB</p>
                        </div>
                        <div class="button-group">
                            <button class="action-btn copy-btn">
                                <i class="fas fa-copy"></i> Copy Summary
                            </button>
                            <button class="action-btn download-btn">
                                <i class="fas fa-download"></i> Download Summary
                            </button>
                        </div>
                    </div>`;

                // Add copy button functionality
                const copyButton = summaryResult.querySelector('.copy-btn');
                copyButton.onclick = () => {
                    navigator.clipboard.writeText(summaryText)
                        .then(() => {
                            copyButton.innerHTML = '<i class="fas fa-check"></i> Copied!';
                            setTimeout(() => {
                                copyButton.innerHTML = '<i class="fas fa-copy"></i> Copy Summary';
                            }, 2000);
                        });
                };

                // Add download button functionality
                const downloadButton = summaryResult.querySelector('.download-btn');
                downloadButton.onclick = () => {
                    // Create blob from summary text
                    const blob = new Blob([summaryText], { type: 'text/plain' });
                    const url = window.URL.createObjectURL(blob);
                    
                    // Create temporary link and trigger download
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `summary_${file.name.split('.')[0]}.txt`;
                    document.body.appendChild(a);
                    a.click();
                    
                    // Cleanup
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);

                    // Show success message
                    downloadButton.innerHTML = '<i class="fas fa-check"></i> Downloaded!';
                    setTimeout(() => {
                        downloadButton.innerHTML = '<i class="fas fa-download"></i> Download Summary';
                    }, 2000);
                };
            })
            .catch(error => {
                console.error('Error:', error);
                summaryResult.innerHTML = `
                    <div class="error">
                        <p>Error: ${error.message}</p>
                        <p>Please try again or contact support if the issue persists.</p>
                    </div>`;
            });
        }
    }

    function handleConversionFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('format', convertFormat.value);

        if (conversionResult) {
            conversionResult.style.display = 'block';
            conversionResult.innerHTML = '<div class="loading"></div>';

            fetch('http://localhost:8080/convert', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                conversionResult.innerHTML = `
                    <div class="result-content">
                        <p>${data.result}</p>
                    </div>`;
            })
            .catch(error => {
                console.error('Error:', error);
                conversionResult.innerHTML = `
                    <div class="error">
                        <p>Error: ${error.message}</p>
                        <p>Please ensure the server is running:</p>
                        <ol>
                            <li>Open terminal/command prompt</li>
                            <li>Navigate to your project directory</li>
                            <li>Run: python server.py</li>
                        </ol>
                    </div>`;
            });
        }
    }

    function handleYouTubeUrl() {
        const youtubeUrl = document.getElementById('youtubeUrl').value;
        const youtubeResult = document.getElementById('youtubeResult');
        
        if (!youtubeUrl) {
            youtubeResult.innerHTML = `
                <div class="error">
                    <p>Please enter a YouTube URL</p>
                </div>`;
            return;
        }

        // Show loading state
        youtubeResult.style.display = 'block';
        youtubeResult.innerHTML = `
            <div class="result-content">
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Fetching and analyzing video transcript...</p>
                </div>
            </div>`;

        // Create form data
        const formData = new FormData();
        formData.append('url', youtubeUrl);

        // Make API call
        fetch('http://localhost:8080/youtube', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Received summary data:', data);
            
            // Update the UI with the summary and transcript
            youtubeResult.innerHTML = `
                <div class="result-content">
                    <h3>Video Summary</h3>
                    <div class="tabs">
                        <button class="tab-btn active" data-tab="summary">Summary</button>
                        <button class="tab-btn" data-tab="transcript">Full Transcript</button>
                    </div>
                    <div class="tab-content">
                        <div id="summary-tab" class="tab-pane active">
                            <div class="summary-container">
                                <div class="summary-text">
                                    ${data.summary}
                                </div>
                            </div>
                        </div>
                        <div id="transcript-tab" class="tab-pane">
                            <div class="transcript-container">
                                <div class="transcript-text">
                                    ${data.original_transcript}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="summary-actions">
                        <button class="action-btn copy-btn" title="Copy to clipboard">
                            <i class="fas fa-copy"></i> Copy Summary
                        </button>
                        <button class="action-btn download-btn" title="Download summary">
                            <i class="fas fa-download"></i> Download Summary
                        </button>
                    </div>
                    <div class="video-info">
                        <p>Processed URL: ${youtubeUrl}</p>
                        <a href="https://youtube.com/watch?v=${data.video_id}" target="_blank" class="video-link">
                            <i class="fab fa-youtube"></i> Watch Video
                        </a>
                    </div>
                </div>`;

            // Add tab switching functionality
            const tabs = youtubeResult.querySelectorAll('.tab-btn');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    // Add active class to clicked tab
                    tab.classList.add('active');
                    // Show corresponding content
                    const tabId = tab.dataset.tab;
                    youtubeResult.querySelectorAll('.tab-pane').forEach(pane => {
                        pane.classList.remove('active');
                    });
                    youtubeResult.querySelector(`#${tabId}-tab`).classList.add('active');
                });
            });

            // Add copy functionality
            const copyBtn = youtubeResult.querySelector('.copy-btn');
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(data.summary).then(() => {
                    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy Summary';
                    }, 2000);
                });
            });

            // Add download functionality
            const downloadBtn = youtubeResult.querySelector('.download-btn');
            downloadBtn.addEventListener('click', () => {
                const summaryText = `Summary:\n${data.summary}\n\nFull Transcript:\n${data.original_transcript}`;
                const blob = new Blob([summaryText], { type: 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `youtube_summary_${data.video_id}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                downloadBtn.innerHTML = '<i class="fas fa-check"></i> Downloaded!';
                setTimeout(() => {
                    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Summary';
                }, 2000);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            youtubeResult.innerHTML = `
                <div class="result-content">
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <p>Error: ${error.message}</p>
                        <p>Please ensure:</p>
                        <ul>
                            <li>You've entered a valid YouTube URL</li>
                            <li>The video has subtitles/captions enabled</li>
                            <li>The video is publicly accessible</li>
                        </ul>
                    </div>
                </div>`;
        });
    }

    // Error handling function
    function showError(message) {
        alert(message);
    }

    // Active navigation link highlighting
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-links a');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= sectionTop - 60) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').substring(1) === current) {
                link.classList.add('active');
            }
        });
    });

    // Add smooth scrolling for the "Get Started" button
    document.querySelector('.cta-button').addEventListener('click', function(e) {
        e.preventDefault();
        const toolsSection = document.querySelector('#tools');
        if (toolsSection) {
            toolsSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });

    // Add these styles to your CSS
    const styles = `
        .summary-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
            white-space: pre-wrap;
            line-height: 1.5;
            max-height: 300px;
            overflow-y: auto;
        }

        .metadata {
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 10px;
        }

        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }

        .action-btn {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .action-btn:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }

        .action-btn i {
            font-size: 14px;
        }

        .copy-btn {
            background-color: #2196F3;
        }

        .copy-btn:hover {
            background-color: #1976D2;
        }

        .download-btn {
            background-color: #4CAF50;
        }

        .download-btn:hover {
            background-color: #45a049;
        }

        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: #666;
        }

        .loading::after {
            content: '';
            width: 20px;
            height: 20px;
            margin-left: 10px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background-color: #fff3f3;
            border: 1px solid #ffcdd2;
            border-radius: 4px;
            padding: 15px;
            color: #d32f2f;
            margin: 10px 0;
        }

        .result-content {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
        }

        .result-content h4 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }

        .tab-btn {
            background-color: #f0f0f0;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .tab-btn.active {
            background-color: #4CAF50;
            color: white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .summary-container {
            margin-bottom: 15px;
        }

        .summary-text {
            white-space: pre-wrap;
            line-height: 1.5;
            max-height: 300px;
            overflow-y: auto;
        }

        .transcript-container {
            margin-bottom: 15px;
        }

        .transcript-text {
            white-space: pre-wrap;
            line-height: 1.5;
            max-height: 300px;
            overflow-y: auto;
        }

        .video-info {
            margin-top: 15px;
            font-size: 0.9em;
            color: #6c757d;
        }

        .video-link {
            color: #007bff;
            text-decoration: none;
        }

        .video-link:hover {
            text-decoration: underline;
        }
    `;

    // Add styles to the document
    const styleSheet = document.createElement("style");
    styleSheet.textContent = styles;
    document.head.appendChild(styleSheet);
}); 