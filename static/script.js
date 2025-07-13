// script.js

// Function to create emojis
function createEmoji() {
    const emojis = ["😄", "😍", "😂", "😊", "😎", "🥳", "😇", "🤩", "😋", "🤗"];
    const numEmojis = Math.floor(Math.random() * 4) + 1; // Randomly choose between 1 to 4 emojis
    for (let i = 0; i < numEmojis; i++) {
        const emoji = emojis[Math.floor(Math.random() * emojis.length)];
        
        const emojiElement = document.createElement("span");
        emojiElement.innerHTML = emoji;
        emojiElement.classList.add("emoji");

        const randomX = Math.random() * window.innerWidth;
        emojiElement.style.left = randomX + "px";
        document.getElementById("emoji-container").appendChild(emojiElement);

        animateEmoji(emojiElement);
    }
}

// Function to animate emojis
function animateEmoji(emojiElement) {
    let posY = window.innerHeight;
    const animateInterval = setInterval(() => {
        posY -= 2; // Decrease the value to increase speed
        emojiElement.style.top = posY + "px";
        if (posY < 0) {
            clearInterval(animateInterval);
            emojiElement.remove();
        }
    }, 20); // Increase the interval to decrease speed
}

// Function to create emojis at intervals
function startEmojiFlow() {
    setInterval(() => {
        createEmoji();
    }, 3000); // Increase the interval to decrease frequency
}

// Start emoji flow when the page is loaded
window.onload = startEmojiFlow;
