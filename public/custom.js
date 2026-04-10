// Chainlit Autoscroll Disabler
// Prevents automatic scrolling to bottom during message streaming
(function() {
    'use strict';
    
    // Store the original scrollTop property descriptor
    const originalDescriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollTop');
    
    if (!originalDescriptor) {
        console.error('Could not find original scrollTop descriptor');
        return;
    }
    
    console.log('Installing Chainlit autoscroll disabler...');
    
    // Override the scrollTop property on all DOM elements
    Object.defineProperty(Element.prototype, 'scrollTop', {
        set: function(value) {
            // Detect autoscroll pattern: scrollTop being set to scrollHeight
            if (this.scrollHeight && value === this.scrollHeight) {
                console.log('Blocked Chainlit autoscroll to bottom');
                // Block the scroll by returning without setting the value
                return;
            }
            
            // For all other scroll operations, use the original setter
            originalDescriptor.set.call(this, value);
        },
        
        get: function() {
            return originalDescriptor.get.call(this);
        },
        
        configurable: true,
        enumerable: true
    });
    
    console.log('Autoscroll disabler active');
})();

// Set custom favicon
(function() {
    const existingFavicons = document.querySelectorAll('link[rel="icon"], link[rel="shortcut icon"]');
    existingFavicons.forEach(f => f.remove());

    const favicon = document.createElement('link');
    favicon.rel = 'icon';
    favicon.type = 'image/png';
    favicon.href = '/public/ANL-triangle.png';
    document.head.appendChild(favicon);
})();

const observer = new MutationObserver(() => {
    // Change "Email address" label to "ANL Username"
    const labels = document.querySelectorAll('label');
    labels.forEach(label => {
        if (label.textContent.includes('Email') || label.textContent.includes('email')) {
            label.textContent = 'ANL Username';
        }
    });

    // Clear email placeholder and change input type
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        if (input.type === 'email' || 
            (input.placeholder && input.placeholder.includes('@'))) {
            input.type = 'text';
            input.placeholder = '';
            input.removeAttribute('placeholder');
            input.setAttribute('placeholder', '');
        }
    });

    // Change login title and add styled logo
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, span, div');
    headings.forEach(el => {
        if (el.textContent.trim() === 'Login to access the app' || 
            el.textContent.trim() === 'Login to access XPCS LLM Assistant') {
            
            // Style the title
            el.innerHTML = '<span style="font-size: 14px; color: #888; font-weight: 400; letter-spacing: 0.5px; text-transform: uppercase; display: block; margin-bottom: -2px;">Login to access</span><span style="font-size: 22px; font-weight: 600; color: #fff; letter-spacing: 0.3px; display: block;">XPCS LLM Assistant</span>';
            el.style.cssText = 'text-align: center; line-height: 1.2; margin-bottom: 1px;';


            // Add Argonne logo above the title if not already added
            if (!document.getElementById('anl-login-logo')) {
                const logoContainer = document.createElement('div');
                logoContainer.id = 'anl-login-logo';
                logoContainer.style.cssText = 'text-align: center; margin-bottom: 0px; padding: 5px 0;';

                const logo = document.createElement('img');
                logo.src = '/public/ANL-logo-white.png';
                logo.alt = 'Argonne National Laboratory';
                logo.style.cssText = 'width: 280px; max-width: 90%;';

                logoContainer.appendChild(logo);
                el.parentNode.insertBefore(logoContainer, el);
            }
        }
    });

    // If on login page, hide all images except ANL logo
    const loginForm = document.querySelector('form');
    if (loginForm) {
        document.querySelectorAll('img').forEach(img => {
            if (img.id !== 'anl-login-logo' && 
                !img.closest('#anl-login-logo')) {
                img.style.display = 'none';
            }
        });
    }

});

observer.observe(document.body, { childList: true, subtree: true });

