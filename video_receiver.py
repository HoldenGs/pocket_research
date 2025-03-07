import socket, cv2, numpy as np

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(65535)
    np_data = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if frame is not None:
        cv2.imshow("Received", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

sock.close()
cv2.destroyAllWindows()