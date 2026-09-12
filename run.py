import sys


def main():
    from application.application import Application

    app = Application.instance()

    app.run()


if __name__ == "__main__":
    sys.exit(main())
