from data.mock.generate_users import generate_users

def main():
    users = generate_users()
    print(users.head())

if __name__ == "__main__":
    main()