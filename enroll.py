"""
enroll.py — CLI tool to enroll new persons into the JARVIS face database.

Usage:
  # Enroll from a single image file
  python enroll.py --image data/known_faces/alice.jpg \
                   --id EMP-001 --name "Alice Chen" \
                   --department Engineering --role "Senior Engineer"

  # Interactive webcam capture
  python enroll.py --webcam \
                   --id EMP-002 --name "Bob Singh" \
                   --department Security --role "Guard"

  # List enrolled persons
  python enroll.py --list

  # Delete a person
  python enroll.py --delete EMP-001
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np

import config
from modules.db import PersonDB
from modules.face_engine import FaceEngine

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
)
logger = logging.getLogger("jarvis.enroll")


def get_embedding_from_image(engine: FaceEngine, image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Cannot read image: %s", image_path)
        return None
    matches = engine.process_frame(img)
    if not matches:
        logger.error("No face detected in: %s", image_path)
        return None
    if len(matches) > 1:
        logger.warning("Multiple faces detected — using the largest one.")
        matches.sort(key=lambda m: m.bbox.width * m.bbox.height, reverse=True)
    emb = matches[0].embedding
    if emb is None:
        logger.error("Embedding extraction failed (InsightFace may not be installed).")
    return emb


def get_embedding_from_webcam(engine: FaceEngine) -> np.ndarray | None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("\n📷  Webcam active. Press SPACE to capture, ESC to cancel.\n")

    emb = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        matches = engine.process_frame(frame)
        for m in matches:
            b = m.bbox
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 70), 2)

        cv2.putText(frame, "SPACE = capture   ESC = cancel",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 70), 1)
        cv2.imshow("JARVIS Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == 32: # SPACE
            if matches:
                emb = matches[0].embedding
                print("✅  Face captured.")
                break
            else:
                print("⚠️  No face in frame — try again.")

    cap.release()
    cv2.destroyAllWindows()
    return emb


def cmd_list(db: PersonDB):
    persons = db.list_persons()
    if not persons:
        print("No persons enrolled.")
        return
    print(f"\n{'ID':<12} {'Name':<22} {'Department':<18} {'Role'}")
    print("─" * 70)
    for p in persons:
        print(f"{p['person_id']:<12} {p['name']:<22} {p['department']:<18} {p.get('role','')}")
    print()


def cmd_delete(db: PersonDB, person_id: str):
    if db.delete_person(person_id):
        print(f"✅  Deleted person '{person_id}'.")
    else:
        print(f"⚠️  Person '{person_id}' not found.")


def main():
    parser = argparse.ArgumentParser(description="JARVIS Face Enrollment Tool")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image",   help="Path to a face image")
    group.add_argument("--webcam",  action="store_true", help="Capture from webcam")
    group.add_argument("--list",    action="store_true", help="List all enrolled persons")
    group.add_argument("--delete",  metavar="PERSON_ID",  help="Delete a person by ID")

    parser.add_argument("--id",         dest="person_id",  help="Unique person ID (e.g. EMP-001)")
    parser.add_argument("--name",       help="Full name")
    parser.add_argument("--department", default="Unknown", help="Department")
    parser.add_argument("--role",       default="",        help="Job role / title")

    args = parser.parse_args()

    db = PersonDB()
    if not db.is_connected():
        print("❌  Cannot connect to MongoDB. Check your .env MONGO_URI setting.")
        sys.exit(1)

    # ── List ──
    if args.list:
        cmd_list(db)
        return

    # ── Delete ──
    if args.delete:
        cmd_delete(db, args.delete)
        return

    # ── Enroll ──
    if not args.person_id or not args.name:
        parser.error("--id and --name are required for enrollment.")

    engine = FaceEngine(db)   # loads model

    if args.image:
        emb = get_embedding_from_image(engine, args.image)
    elif args.webcam:
        emb = get_embedding_from_webcam(engine)
    else:
        parser.error("Specify --image or --webcam to enroll.")
        return

    if emb is None:
        print("❌  Enrollment failed — no valid embedding extracted.")
        sys.exit(1)

    ok = db.enroll_person(
        person_id  = args.person_id,
        name       = args.name,
        department = args.department,
        role       = args.role,
        embedding  = emb,
        photo_path = args.image,
    )

    if ok:
        print(f"\n✅  Enrolled: {args.name} (ID={args.person_id}, Dept={args.department})\n")
    else:
        print("❌  Database write failed.")


if __name__ == "__main__":
    main()
