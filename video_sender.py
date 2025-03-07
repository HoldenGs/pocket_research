import cv2
import socket

server_address = ('192.168.1.172', 5005)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Compress frame to JPEG
    ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        continue
    data = buf.tobytes()
    sock.sendto(data, server_address)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
sock.close()