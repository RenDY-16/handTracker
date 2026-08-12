[DESIGN DIRECTIVE: HIGH-END CINEMATIC AESTHETIC (ZERO ALAY)]

Tolong lakukan audit visual pada `main.py` dan rapikan semua efek filter, HUD, dan gesture indicator agar terlihat modern, elegan, dan profesional (sekelas software streamer / AAA game HUD).

IKUTI ATURAN DESAIN BERIKUT:

1. HILANGKAN LANDMARK MENTAH (NO NEON LINES):
   - Sembunyikan/hapus garis laser tebal dan titik-titik bawaan MediaPipe `mp_drawing.draw_landmarks`.
   - Hanya tampilkan efek jika gestur aktif, dan gunakan indikator yang minimalis (misal: aksen cahaya halus di pergelangan tangan atau ujung jari).

2. PILIHAN WARNA ELEGAND (PALETTE):
   - Jangan gunakan warna murni 100% mencolok (seperti Red #FF0000 atau Green #00FF00 murni).
   - Gunakan warna desaturated / cinematic palette:
     * Spider-Man: Crimson Deep Red & Midnight Blue dengan opacity halus.
     * Venom: Dark Obsidian Slate & Subtle Purple/Black Vignette.
     * HUD/UI: Monokromatik (White 80% opacity, Dark Glass gray).

3. SMOTHERING & TRANSPARANSI (ALPHA BLENDING):
   - Semua overlay (overlay warna, halftone, atau vignette) wajib menggunakan `cv2.addWeighted()` dengan alpha transparansi 0.2 - 0.4 agar wajah/kamera tetap jernih.
   - Gunakan `cv2.LINE_AA` (Anti-Aliased) di setiap pembuatan garis atau teks agar tidak patah-patah/patah piksel.

4. TYPOGRAPHY & HUD:
   - Gunakan font yang bersih dan ukuran kecil-sedang (fontScale 0.4 - 0.6).
   - Buat padding UI yang rapi dengan latar belakang frosted glass hitam transparan.

Lakukan refactor pada bagian rendering visual di `main.py` sekarang juga berdasarkan aturan di atas!