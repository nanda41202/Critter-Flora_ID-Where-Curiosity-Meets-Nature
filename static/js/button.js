document.addEventListener('DOMContentLoaded', function() {
    // Handle logout on Escape key
    document.addEventListener('keypress', function(e) {
        if (e.key === 'Escape') {
            // Use the actual URL path for logout
            window.location.href = '/logout'; 
        }
    });
}); 