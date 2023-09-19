import paho.mqtt.client as mqtt
import json
from pprint import pprint


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("coordinates")
        print("Subscribed to subscriber topic")
    else:
        print("Connection failed")


def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8")
    data = payload
    pprint(json.loads(data))


# mqttBroker = "mqtt.eclipseprojects.io"
mqttBroker = "localhost"

client = mqtt.Client("Boxes", clean_session=False)
client.on_connect = on_connect
client.on_message = on_message

client.connect(mqttBroker, 1884)

client.loop_forever()
