function handleImageUpload(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    fetch('/upload_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultBox = document.getElementById('detection-results');
        resultBox.innerHTML = `<img src="data:image/jpeg;base64,${data.image}" />`;
        if (data.high_risk_detected) {
            playAlert();
        }
        document.getElementById('back-button').style.display = 'block';
        document.getElementById('resultModal').style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
}

function handleVideoUpload(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultBox = document.getElementById('detection-results');
        resultBox.innerHTML = `<video src="data:video/mp4;base64,${data.video}" controls></video>`;
        if (data.high_risk_detected) {
            playAlert();
        }
        document.getElementById('back-button').style.display = 'block';
        document.getElementById('resultModal').style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
}

function playAlert() {
    const alertSound = document.getElementById('alert-sound');
    alertSound.play();
    document.getElementById('stop-alert').style.display = 'block';
}

function stopAlert() {
    fetch('/stop_alarm', { method: 'POST' })
    .then(response => response.json())
    .then(() => {
        document.getElementById('stop-alert').style.display = 'none';
        const alertSound = document.getElementById('alert-sound');
        alertSound.pause();
        alertSound.currentTime = 0;
    });
}

document.getElementById('stop-alert').addEventListener('click', stopAlert);
