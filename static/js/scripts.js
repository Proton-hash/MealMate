/* static/js/scripts.js */
document.addEventListener('DOMContentLoaded', function () {
    const botText = document.getElementById('bot-text');
    const typingDots = document.getElementById('typing-dots');

    if (botText) {
        const text = botText.innerText;
        botText.innerText = '';
        typingDots.style.display = 'inline-block';

        let i = 0;
        const typingSpeed = 50; // Adjust typing speed here (ms per character)

        const typeWriter = () => {
            if (i < text.length) {
                botText.innerText += text.charAt(i);
                i++;
                setTimeout(typeWriter, typingSpeed);
            } else {
                typingDots.style.display = 'none';
            }
        };

        setTimeout(typeWriter, 500); // Delay before typing starts
    }
});
