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
