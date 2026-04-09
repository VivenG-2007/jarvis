from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from modules.db import PersonDB
from modules.face_engine import FaceEngine


def get_embedding_from_image(engine: FaceEngine, image_path: str) -> np.ndarray | None:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return None
    matches = engine.process_frame(frame)
    if not matches:
        print("No face detected in that image.")
        return None
    matches.sort(key=lambda item: item.bbox.width * item.bbox.height, reverse=True)
    return matches[0].embedding


def get_embedding_from_webcam(engine: FaceEngine) -> np.ndarray | None:
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture a face, or ESC to cancel.")
    embedding = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        matches = engine.process_frame(frame)
        for match in matches:
            cv2.rectangle(frame, (match.bbox.x1, match.bbox.y1), (match.bbox.x2, match.bbox.y2), (0, 255, 120), 2)
        cv2.imshow("JARVIS Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32 and matches:
            embedding = matches[0].embedding
            break
    cap.release()
    cv2.destroyAllWindows()
    return embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Local JARVIS face enrollment")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", help="Path to a face image")
    group.add_argument("--webcam", action="store_true", help="Capture a face from the webcam")
    group.add_argument("--list", action="store_true", help="List enrolled people")
    group.add_argument("--delete", metavar="PERSON_ID", help="Delete a person by ID")

    parser.add_argument("--id", dest="person_id", help="Unique person ID")
    parser.add_argument("--name", help="Full name")
    parser.add_argument("--department", default="Unknown")
    parser.add_argument("--role", default="")
    parser.add_argument("--notes", default="")

    args = parser.parse_args()
    db = PersonDB()

    if args.list:
        for person in db.list_persons():
            print(f"{person['person_id']:10}  {person['name']:20}  {person['department']:16}  {person.get('role', '')}")
        return

    if args.delete:
        deleted = db.delete_person(args.delete)
        print("Deleted." if deleted else "Person not found.")
        return

    if not args.person_id or not args.name:
        parser.error("--id and --name are required for enrollment")

    engine = FaceEngine(db)
    embedding = get_embedding_from_image(engine, args.image) if args.image else get_embedding_from_webcam(engine)
    if embedding is None:
        print("Enrollment failed.")
        sys.exit(1)

    db.enroll_person(
        person_id=args.person_id,
        name=args.name,
        department=args.department,
        role=args.role,
        embedding=embedding,
        photo_path=args.image,
        notes=args.notes,
    )
    print(f"Enrolled {args.name} as {args.person_id}.")


if __name__ == "__main__":
    main()
