# GitHub Pages project page

This `docs/` folder is ready to be used as a GitHub Pages static project page.

## Recommended repository layout

```text
repository-root/
├── code/
├── data/
├── docs/
│   ├── index.html
│   ├── assets/
│   │   ├── css/style.css
│   │   ├── js/main.js
│   │   ├── images/fig1.png ... fig6.png
│   │   ├── paper/paper.pdf
│   │   └── videos/
│   └── README.md
└── README.md
```

## GitHub Pages deployment

1. Put this `docs/` folder at the root of your GitHub repository.
2. Open **Settings → Pages**.
3. Set **Source** to **Deploy from a branch**.
4. Select the `main` branch and `/docs` folder.
5. Save. GitHub will generate a project page URL after deployment.

## Items to update before public release

- Replace `Author Name · Institution` in `index.html` with final author and affiliation information.
- Replace the BibTeX placeholder after the paper is accepted or formally released.
- Add actual demo videos to `assets/videos/` and update the Video Demo section.
- Add the final code repository link to the button row if the project page is hosted separately from the code.
