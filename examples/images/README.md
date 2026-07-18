# Example images (real photos)

Real break-room photos for testing the vision path. Resized to ≤1280 px
(the model doesn't need more, and it keeps `ask_image` fast on CPU).

| File | Scene | Good test questions |
| --- | --- | --- |
| `snack_wall_1.jpg` | Wall-mounted wire racks + baskets with cookies, chips, fruit snacks, candy | "Is any snack basket empty?" / "Are there any cookies left?" |
| `snack_wall_2.jpg` | Same snack wall, different angle — includes a visibly empty basket | "Is any snack basket empty?" (expected: yes) |
| `supply_shelf.jpg` | Amenities shelf: air freshener, mints, wipes, first-aid packets | "Is the shelf out of wipes?" / "Are the bins stocked?" |

## Use them

```bash
# after seed_demo.py and with the app + model running:
cp examples/images/snack_wall_2.jpg inbox/breakroom_cam/
curl -X POST localhost:8000/api/cycle
```

Then check the Alerts panel / evaluations log for the model's answer and
reasoning.
