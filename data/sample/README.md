Sample registry entries live in `data/memory/persons.json`.

Those records use zero embeddings as placeholders so the UI and local memory format are easy to inspect, but they will not produce real matches until you enroll actual faces with:

```bash
python enroll.py --image path/to/photo.jpg --id EMP-100 --name "Your Name" --department Engineering
```
