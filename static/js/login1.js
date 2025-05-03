// Enhanced version with error message handling
document.addEventListener("DOMContentLoaded", function() {
    // Check URL parameters for status messages
    const urlParams = new URLSearchParams(window.location.search);
    
    // Handle URL parameter messages
    if (urlParams.get("registration") === "success") {
        showAlert("Registration successful! Please login.", "success");
    }
    
    if (urlParams.get("logout") === "success") {
        showAlert("You have been logged out successfully.", "info");
    }
    
    // Handle flash messages and convert them to custom alerts
    const flashMessages = document.querySelectorAll(".alert");
    if (flashMessages.length > 0) {
        flashMessages.forEach(message => {
            const messageText = message.textContent.trim();
            let messageType = "info";
            
            if (message.classList.contains("alert-success")) {
                messageType = "success";
            } else if (message.classList.contains("alert-error")) {
                messageType = "error";
            }
            
            // Remove the original flash message
            message.remove();
            
            // Create a custom alert instead
            showAlert(messageText, messageType);
        });
    }

    // Handle forgot password form submission
    const forgotForm = document.querySelector('form[action="/forgot-password"]');
    if (forgotForm) {
        forgotForm.addEventListener("submit", function(e) {
            e.preventDefault();
            const loginId = this.querySelector('input[name="login_id"]').value;
            
            if (!loginId) {
                showAlert("Please enter your email or username", "error");
                return;
            }
            
            this.submit();
        });
    }

    // Function to show custom alerts
    function showAlert(message, type = "info") {
        const alertDiv = document.createElement("div");
        alertDiv.className = `alert alert-${type} fade-in`;
        alertDiv.textContent = message;
        
        const container = document.querySelector(".container");
        if (container) {
            // Add before the form element
            const form = container.querySelector("form");
            if (form) {
                container.insertBefore(alertDiv, form);
            } else {
                container.insertBefore(alertDiv, container.firstChild);
            }
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                alertDiv.classList.add("fade-out");
                setTimeout(() => alertDiv.remove(), 500);
            }, 5000);
        }
    }
    
}); 