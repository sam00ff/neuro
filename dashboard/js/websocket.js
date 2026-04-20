export class BrainWebSocket {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.onInit = null;
        this.onState = null;
        this.onResponse = null;
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('[WS] Connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);

                if (msg.type === 'init' && this.onInit) {
                    this.onInit(msg.positions);

                } else if (msg.type === 'state' && this.onState) {
                    this.onState(msg.data);

                } else if (msg.type === 'response' && this.onResponse) {
                    this.onResponse(msg.data);
                }

            } catch (e) {
                console.error('[WS] Parse error:', e);
            }
        };
    }

    sendTextInput(text) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'text_input', text }));
        }
    }
}