const chatbox = document.getElementById('chatbox');
const message = document.getElementById('message');
const send = document.getElementById('send');
const infoText = document.getElementById('info-text');

function botSay(botMessage) {
    const botDiv = document.createElement('div');
    botDiv.className = 'message bot';
    botDiv.innerHTML = `<strong>magnetron:</strong> ${botMessage}`;
    chatbox.appendChild(botDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

send.addEventListener('click', function() {
    const userMessage = message.value.trim();
    if (userMessage) {
        const userDiv = document.createElement('div');
        userDiv.className = 'message user';
        userDiv.innerHTML = `<strong>You:</strong> ${userMessage}`;
        chatbox.appendChild(userDiv);

        message.value = "";

        fetch(`/api/v1/bot_response?message=${encodeURIComponent(userMessage)}`)
            .then(response => response.text())
            .then(botMessage => {
                botSay(botMessage);
            })
            .catch(error => {
                console.error('Error fetching response:', error);
            });

        fetch(`/api/v1/system_info`)
            .then(response => response.text())
            .then(infoString => {
                infoText.innerHTML = infoString;
            })
            .catch(error => {
                console.error('Error fetching response:', error);
            });
    }
});

fetch(`/api/v1/system_info`)
    .then(response => response.text())
    .then(infoString => {
        infoText.innerHTML = infoString;
    })
    .catch(error => {
        console.error('Error fetching response:', error);
    });

message.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') send.click();
});

botSay('Hello! I\'m an AI trained to solve the XOR function. Enter me two space separated numbers in {0, 1} and press enter. Like 1 0. Or 1 1.')
