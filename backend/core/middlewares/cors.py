from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3005",
    "https://quivr.app",
    "https://www.quivr.app",
    "http://quivr.app",
    "http://www.quivr.app",
    "https://platform-jet.vercel.app",
    "platform-*-liamvmurphy.vercel.app",
    "*",
]


def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
