// Replace with your text endpoint URL

let isPlaying = false;
let loopPromise;

async function fetchText() {
    try {
        const response = await fetch('/get_text');
        const data = await response.json();
        
        return data
    } catch (error) {
        //console.error('Error fetching text:', error);
        return null;
    }
}

async function getVoiceAudio(role, text) {
    try {
        const response = await fetch('/api/elevenlabs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({  role, text })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const blob = await response.blob();
        return URL.createObjectURL(blob);
    } catch (error) {
        console.error('Error fetching voice audio:', error);
        return null;
    }
}

async function playAudio(audioUrl) {
    const audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = audioUrl;
    try {
        await audioPlayer.play();
        console.log('Audio is playing');
    } catch (error) {
        console.error('Error playing audio:', error);
    }
}

document.getElementById('start-button').onclick = () => {
    if (!isPlaying) {
        isPlaying = true;
        loopPromise = startLoop();
    }
};

document.getElementById('stop-button').onclick = () => {
    isPlaying = false;
};

async function startLoop() {
    const displayText = document.getElementById('display-text');
    const audioPlayer = document.getElementById('audio-player');
    while (isPlaying) {
        const data = await fetchText();
        if (!data) {
            displayText.textContent = 'Failed to retrieve text. Retrying...';
            await new Promise(resolve => setTimeout(resolve, 5000));
            continue;
        }

        displayText.textContent = `Now Playing: ${data.text}`;

        const audioUrl = await getVoiceAudio(data.role, data.text);
        if (!audioUrl) {
            console.log('Failed to get audio. Waiting before retrying...');
            await new Promise(resolve => setTimeout(resolve, 5000));
            continue;
        }

        await playAudio(audioUrl);

        // Wait for the audio to finish before next iteration
        await new Promise(resolve => {
            audioPlayer.onended = resolve;
            if (!isPlaying) {
                audioPlayer.pause();
                resolve();
            }
        });
    }
}

// Start the loop when the page loads
window.onload = startLoop;
