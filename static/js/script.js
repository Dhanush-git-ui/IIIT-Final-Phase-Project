let mediaRecorder;
let audioChunks = [];
let audioBlob = null;

let audioContext;
let processor;
let input;
let globalStream;
let leftchannel = [];
let recordingLength = 0;
let sampleRate = 16000;

// --- AUDIO RECORDING (MONO 16KHz WAV WRITER) ---
async function startRecording() {
    leftchannel = [];
    recordingLength = 0;
    audioBlob = null;
    document.getElementById('predictSpeechBtn').disabled = true;
    document.getElementById('predictFusionBtn').disabled = true;
    
    try {
        globalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Resample to 16000Hz (exact match for speech model)
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: sampleRate
        });
        
        input = audioContext.createMediaStreamSource(globalStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = function(e) {
            let left = e.inputBuffer.getChannelData(0);
            leftchannel.push(new Float32Array(left));
            recordingLength += 4096;
        };
        
        input.connect(processor);
        processor.connect(audioContext.destination);
        
        setupVisualizer(globalStream);
        
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        
    } catch (err) {
        console.error("Recording initialization failed: ", err);
        alert("Could not access microphone. Ensure permissions are allowed.");
    }
}

function stopRecording() {
    if (processor) {
        processor.disconnect();
        input.disconnect();
    }
    if (globalStream) {
        globalStream.getTracks().forEach(track => track.stop());
        globalStream = null;
    }
    if (audioContext) {
        audioContext.close();
    }
    
    let result = new Float32Array(recordingLength);
    let offset = 0;
    for (let i = 0; i < leftchannel.length; i++) {
        result.set(leftchannel[i], offset);
        offset += leftchannel[i].length;
    }
    
    audioBlob = bufferToWav(result, sampleRate);
    
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = document.getElementById('audioPlayback');
    audio.src = audioUrl;
    audio.style.display = 'block';
    
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('predictSpeechBtn').disabled = false;
    document.getElementById('predictFusionBtn').disabled = false;
}

// Standard PCM WAV Header builder
function bufferToWav(buffer, sampleRate) {
    let bufferArr = new ArrayBuffer(44 + buffer.length * 2);
    let view = new DataView(bufferArr);
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + buffer.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); // 16-bit
    writeString(view, 36, 'data');
    view.setUint32(40, buffer.length * 2, true);
    
    let index = 44;
    for (let i = 0; i < buffer.length; i++) {
        let s = Math.max(-1, Math.min(1, buffer[i]));
        view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        index += 2;
    }
    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// --- LIVE WAVEFORM VISUALIZATION ---
function setupVisualizer(stream) {
    const visualizerCtx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = visualizerCtx.createAnalyser();
    const source = visualizerCtx.createMediaStreamSource(stream);
    source.connect(analyser);
    
    analyser.fftSize = 64;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const canvas = document.getElementById('visualizerCanvas');
    const canvasCtx = canvas.getContext('2d');
    
    function draw() {
        if (!globalStream) {
            canvasCtx.fillStyle = 'rgba(15, 16, 22, 1)';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
            return;
        }
        
        requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);
        canvasCtx.fillStyle = 'rgba(15, 16, 22, 0.2)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        let barWidth = (canvas.width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;
        
        for(let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 2.5;
            let gradient = canvasCtx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
            gradient.addColorStop(0, '#10B981'); // Emerald (Secondary)
            gradient.addColorStop(0.5, '#38BDF8'); // Sky Blue (Accent)
            gradient.addColorStop(1, '#22D3EE'); // Cyan (Primary)
            
            canvasCtx.fillStyle = gradient;
            canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 3, barHeight);
            x += barWidth;
        }
    }
    draw();
}

// --- BACKEND API COMMUNICATIONS ---
async function predictText() {
    const text = document.getElementById('textInput').value;
    if(!text.trim()) return alert("Please type something first!");
    
    document.getElementById('textResult').innerText = "Analyzing...";
    document.getElementById('textResult').style.opacity = "0.5";

    try {
        const response = await fetch('/predict_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        
        const resultDiv = document.getElementById('textResult');
        resultDiv.innerText = data.emotion || data.error;
        resultDiv.style.opacity = "1";
    } catch (err) {
        document.getElementById('textResult').innerText = "Error";
        document.getElementById('textResult').style.opacity = "1";
    }
}

async function predictSpeech() {
    if (!audioBlob) return;
    document.getElementById('speechResult').innerText = "Processing Audio...";
    document.getElementById('speechResult').style.opacity = "0.5";

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
        const response = await fetch('/predict_speech', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        const resultDiv = document.getElementById('speechResult');
        resultDiv.innerText = data.emotion || data.error;
        resultDiv.style.opacity = "1";
    } catch (err) {
        document.getElementById('speechResult').innerText = "Error";
        document.getElementById('speechResult').style.opacity = "1";
    }
}

async function predictFusion() {
    const text = document.getElementById('textInput').value;
    if(!text.trim()) return alert("Please input text in the 'Text Analysis' card first!");
    if (!audioBlob) return alert("Please record a voice note first!");
    
    document.getElementById('fusionResult').innerText = "Calculating Fusion...";
    document.getElementById('fusionResult').style.opacity = "0.5";
    document.getElementById('fusionConfidence').innerText = "";

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    formData.append('text', text);

    try {
        const response = await fetch('/predict_fusion', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        const resultDiv = document.getElementById('fusionResult');
        resultDiv.innerText = data.emotion || data.error;
        resultDiv.style.opacity = "1";
        
        if(data.confidence) {
            document.getElementById('fusionConfidence').innerText = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
        }
    } catch (err) {
        document.getElementById('fusionResult').innerText = "Error";
        document.getElementById('fusionResult').style.opacity = "1";
    }
}

// --- FILE UPLOAD PROCESSING ---
function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Store in global audioBlob variable
    audioBlob = file;
    
    // Set up audio playback
    const audioUrl = URL.createObjectURL(file);
    const audio = document.getElementById('audioPlayback');
    audio.src = audioUrl;
    audio.style.display = 'block';
    
    // Enable analyze buttons
    document.getElementById('predictSpeechBtn').disabled = false;
    document.getElementById('predictFusionBtn').disabled = false;
}
