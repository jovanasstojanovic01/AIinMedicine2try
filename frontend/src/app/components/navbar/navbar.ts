import { Component, Renderer2, Inject, OnInit } from '@angular/core';
import { CommonModule, DOCUMENT } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterModule, MatToolbarModule, MatButtonModule, MatIconModule],
  templateUrl: './navbar.html',
  styleUrls: ['./navbar.scss'],
})
export class Navbar implements OnInit {
  private readonly THEME_KEY = 'oftascan_theme_mode';
  isDarkTheme = false;

  constructor(
    private renderer: Renderer2,
    @Inject(DOCUMENT) private document: Document,
  ) {}

  ngOnInit(): void {
    // Ako smo u browseru, samo proveravamo da li je index.html već uspešno upalio tamnu temu
    if (typeof window !== 'undefined') {
      this.isDarkTheme = this.document.body.classList.contains('dark-theme');
    }
  }

  toggleTheme(): void {
    this.isDarkTheme = !this.isDarkTheme;

    if (typeof window !== 'undefined') {
      if (this.isDarkTheme) {
        this.renderer.addClass(this.document.body, 'dark-theme');
        localStorage.setItem(this.THEME_KEY, 'dark'); // KLJUČNO: Sada stvarno pamtimo!
      } else {
        this.renderer.removeClass(this.document.body, 'dark-theme');
        localStorage.setItem(this.THEME_KEY, 'light');
      }
    }
  }
}
