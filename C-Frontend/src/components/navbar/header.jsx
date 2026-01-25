import React, { useState } from "react";
import { Link } from "react-router-dom";
import useAuth from "../../hooks/auth/useAuth";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowRightFromBracket } from "@fortawesome/free-solid-svg-icons";
import "./styles.css"; 

const Header = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const { logout } = useAuth();

  const navLinks = [
    { label: "Cart", path: "/cart" },
    { label: "Buy", path: "/buy" },
    { label: "Best Sellers", path: "/best" },
    { label: "Wallet", path: "/wallet" }, 
  ];

  return (
    <header className="header">
      <div className="header-content">

        {/* Hamburger (mobile) */}
        <button
          className="hamburger-btn"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          ☰
        </button>

        {/* Logo + Brand */}
        <h1 className="logo">
          <img
            src="/circle.png"
            alt="GFriends Logo"
            className="logo-image"
          />
          GFriend's Korean Mart
        </h1>

        {/* Navigation */}
        <nav className={`nav-links ${menuOpen ? "open" : ""}`}>
          {navLinks.map((link, index) => (
            <Link
              key={index}
              to={link.path}
              className="nav-link"
              onClick={() => setMenuOpen(false)}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        {/* Logout */}
        <button onClick={logout} className="logout-btn">
          <FontAwesomeIcon
            icon={faArrowRightFromBracket}
            className="logout-icon"
          />
        </button>

      </div>
    </header>
  );
};

export default Header;
