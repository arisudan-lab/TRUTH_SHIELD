const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const fileDetails = document.getElementById('file-details');
const fileName = document.getElementById('file-name');
const progressFill = document.getElementById('progress-fill');
const uploadBtn = document.getElementById('upload-btn');

// Trigger file input when clicking button or box
browseBtn.onclick = () => fileInput.click();
dropZone.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    handleFile(e.target.files[0]);
};

// Drag and Drop listeners
dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
};

dropZone.ondragleave = () => dropZone.classList.remove('active');

dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    handleFile(e.dataTransfer.files[0]);
};

function handleFile(file) {
    if (file) {
        fileName.innerText = file.name;
        document.getElementById('file-size').innerText = (file.size / 1024).toFixed(1) + " KB";
        fileDetails.style.display = 'block';
        progressFill.style.width = '0%';
    }
}

// Simulate Upload
uploadBtn.onclick = () => {
    let progress = 0;
    uploadBtn.disabled = true;
    
    const interval = setInterval(() => {
        progress += 10;
        progressFill.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(interval);
            document.getElementById('status-text').innerText = "Upload Complete!";
            uploadBtn.innerText = "Done";
        }
    }, 200);
};
