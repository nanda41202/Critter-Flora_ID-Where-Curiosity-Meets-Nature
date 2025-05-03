document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.querySelector('form[action="{{ url_for('upload_file') }}"]');
    const fileInput = uploadForm.querySelector('input[type="file"]');
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            // Validate file type if needed
            const validTypes = ['video/mp4', 'video/webm', 'video/ogg'];
            if (!validTypes.includes(file.type)) {
                alert('Please select a valid video file (MP4, WebM, or OGG)');
                this.value = '';
                return;
            }
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files[0]) {
            alert('Please select a file to upload');
            return;
        }
        
        this.submit();
    });

    // Handle logout on Escape key
    document.addEventListener('keypress', function(e) {
        if (e.key === 'Escape') {
            window.location.href = "{{ url_for('logout') }}";
        }
    });
}); 