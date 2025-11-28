import socket

try:
    print(socket.gethostbyname("db.bimbbqivckkagulclmhd.supabase.co"))
except Exception as e:
    print("Socket error:", e)
