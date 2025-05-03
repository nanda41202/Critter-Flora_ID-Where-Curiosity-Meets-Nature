/**
 * Registration Page JavaScript
 * Handles password validation, toggling password visibility, and animation effects
 */

// Toggle password visibility between text and password
function togglePassword(inputId) {
    const passwordInput = document.getElementById(inputId);
    const toggleIcon = inputId === 'password' ? document.getElementById('toggleIcon1') : document.getElementById('toggleIcon2');
    
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        toggleIcon.classList.remove('fa-eye');
        toggleIcon.classList.add('fa-eye-slash');
    } else {
        passwordInput.type = 'password';
        toggleIcon.classList.remove('fa-eye-slash');
        toggleIcon.classList.add('fa-eye');
    }
}

// Real-time password validation
function validatePassword() {
    const password = document.getElementById('password').value;
    const criteriaSection = document.getElementById('password-criteria');
    
    // Show criteria section when user starts typing
    if (password.length > 0) {
        criteriaSection.style.display = 'block';
        // Add a small delay before adding opacity for smooth transition
        setTimeout(() => {
            criteriaSection.style.opacity = '1';
        }, 10);
    } else {
        criteriaSection.style.opacity = '0';
        // Hide after fade-out
        setTimeout(() => {
            criteriaSection.style.display = 'none';
        }, 300);
        return;
    }
    
    // Update each criteria
    const lengthCriteria = document.getElementById('length-criteria');
    const lowercaseCriteria = document.getElementById('lowercase-criteria');
    const uppercaseCriteria = document.getElementById('uppercase-criteria');
    const numberCriteria = document.getElementById('number-criteria');
    const specialCriteria = document.getElementById('special-criteria');
    
    // Check if criteria are met
    const hasLength = password.length >= 8;
    const hasLowercase = /[a-z]/.test(password);
    const hasUppercase = /[A-Z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password);
    
    // Update criteria status
    updateCriteria(lengthCriteria, hasLength);
    updateCriteria(lowercaseCriteria, hasLowercase);
    updateCriteria(uppercaseCriteria, hasUppercase);
    updateCriteria(numberCriteria, hasNumber);
    updateCriteria(specialCriteria, hasSpecial);
    
    // Enable/disable submit button based on all criteria being met
    document.getElementById('submit-btn').disabled = !(hasLength && hasLowercase && hasUppercase && hasNumber && hasSpecial);
}

// Update visual indicators for password criteria
function updateCriteria(element, isMet) {
    if (isMet) {
        element.classList.add('criteria-met');
    } else {
        element.classList.remove('criteria-met');
    }
}

// Check for password match between password and confirm password fields
function checkPasswordMatch() {
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm_password').value;
    const submitBtn = document.getElementById('submit-btn');
    
    if (confirmPassword.length > 0) {
        if (password !== confirmPassword) {
            document.getElementById('password-match').style.display = 'block';
            document.getElementById('password-match').classList.remove('match');
            document.getElementById('password-match').classList.add('no-match');
            document.getElementById('password-match').innerText = 'Passwords do not match';
            submitBtn.disabled = true;
        } else {
            document.getElementById('password-match').style.display = 'block';
            document.getElementById('password-match').classList.remove('no-match');
            document.getElementById('password-match').classList.add('match');
            document.getElementById('password-match').innerText = 'Passwords match';
            // Still check if other criteria are met before enabling submit
            validatePassword();
        }
    } else {
        document.getElementById('password-match').style.display = 'none';
    }
}

// Initialize event listeners when DOM content is loaded
document.addEventListener('DOMContentLoaded', function() {
    const passwordField = document.getElementById('password');
    const confirmPasswordField = document.getElementById('confirm_password');
    const criteriaSection = document.getElementById('password-criteria');
    
    // Initialize password criteria section
    if (criteriaSection) {
        criteriaSection.style.opacity = '0';
        
        // Show password criteria on focus if field has content
        passwordField.addEventListener('focus', function() {
            if (this.value.length > 0) {
                criteriaSection.style.display = 'block';
                setTimeout(() => {
                    criteriaSection.style.opacity = '1';
                }, 10);
            }
        });
        
        // Hide criteria when focus is lost and password field is empty
        passwordField.addEventListener('blur', function() {
            if (this.value.length === 0) {
                criteriaSection.style.opacity = '0';
                // Hide after fade-out
                setTimeout(() => {
                    criteriaSection.style.display = 'none';
                }, 300);
            }
        });
    }
    
    // Add password match checking if confirm password field exists
    if (confirmPasswordField) {
        confirmPasswordField.addEventListener('input', checkPasswordMatch);
        // Also check when password changes
        passwordField.addEventListener('input', function() {
            if (confirmPasswordField.value.length > 0) {
                checkPasswordMatch();
            }
        });
    }
    
    // Animation for alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            alert.classList.add('fade-in');
        }, 100);
        
        setTimeout(function() {
            alert.classList.remove('fade-in');
            alert.classList.add('fade-out');
        }, 5000);
    });
}); 