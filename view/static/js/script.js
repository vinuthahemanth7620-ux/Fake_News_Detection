/**
 * script.js
 * =========
 * Client-side interactivity for the Fake News Detection Application.
 * Handles character counter, input validation, clear button, loading spinner,
 * and quick test sample populating.
 */

document.addEventListener('DOMContentLoaded', () => {
    const newsForm = document.getElementById('news-form');
    const newsText = document.getElementById('news_text');
    const detectBtn = document.getElementById('detect-btn');
    const clearBtn = document.getElementById('clear-btn');
    const charCounter = document.getElementById('char-counter');
    const validationMsg = document.getElementById('validation-msg');
    const formCard = document.getElementById('form-card');
    const loadingSpinner = document.getElementById('loading-spinner');
    const samplePills = document.querySelectorAll('.sample-pill');

    /**
     * Updates character counter, validates text input, and toggles detect button state.
     */
    function handleInputChange() {
        if (!newsText) return;

        const textLength = newsText.value.length;
        const trimmedText = newsText.value.trim();

        // Update character count
        if (charCounter) {
            charCounter.textContent = `${textLength.toLocaleString()} ${textLength === 1 ? 'character' : 'characters'}`;
        }

        // Validate text presence
        if (trimmedText.length === 0) {
            if (detectBtn) detectBtn.disabled = true;
            if (validationMsg) {
                if (textLength > 0) {
                    validationMsg.textContent = 'Please enter valid news article text (only whitespace detected).';
                    validationMsg.classList.remove('d-none');
                } else {
                    validationMsg.classList.add('d-none');
                }
            }
        } else {
            if (detectBtn) detectBtn.disabled = false;
            if (validationMsg) validationMsg.classList.add('d-none');
        }
    }

    /**
     * Clears article input text and resets validation states.
     */
    function handleClear() {
        if (!newsText) return;
        newsText.value = '';
        newsText.focus();
        handleInputChange();
    }

    /**
     * Handles form submission: shows loading spinner animation and disables buttons.
     */
    function handleFormSubmit(e) {
        if (!newsText) return;

        const trimmedText = newsText.value.trim();
        if (!trimmedText) {
            e.preventDefault();
            if (validationMsg) {
                validationMsg.textContent = 'Please paste or type a news article before detecting.';
                validationMsg.classList.remove('d-none');
            }
            return;
        }

        // Show loading spinner animation
        if (formCard) formCard.classList.add('d-none');
        if (loadingSpinner) loadingSpinner.classList.remove('d-none');
    }

    // Register Event Listeners
    if (newsText) {
        newsText.addEventListener('input', handleInputChange);
        newsText.addEventListener('keyup', handleInputChange);
        newsText.addEventListener('change', handleInputChange);
        // Initial state check
        handleInputChange();
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', handleClear);
    }

    if (newsForm) {
        newsForm.addEventListener('submit', handleFormSubmit);
    }

    // Attach quick sample pill handlers
    samplePills.forEach(pill => {
        pill.addEventListener('click', () => {
            const sampleText = pill.getAttribute('data-sample');
            if (sampleText && newsText) {
                newsText.value = sampleText;
                handleInputChange();
                newsText.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    });
});
