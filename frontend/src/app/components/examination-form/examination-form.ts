import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
// Material uvozi
import { MatStepperModule } from '@angular/material/stepper';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatSelectModule } from '@angular/material/select';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';

@Component({
  selector: 'app-examination-form',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatStepperModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatSelectModule,
    MatIconModule,
    MatCardModule,
  ],
  templateUrl: './examination-form.html',
  styleUrls: ['./examination-form.scss'],
})
export class ExaminationForm implements OnInit {
  vrstaPregledaForm!: FormGroup;
  pacijentForm!: FormGroup;
  pregledForm!: FormGroup;

  // Simulacija uploada slika
  slikaLevoIme: string | null = null;
  slikaDesnoIme: string | null = null;

  constructor(
    private fb: FormBuilder,
    private router: Router,
  ) {}

  ngOnInit(): void {
    // Korak 1: Biranje tipa pregleda i pronalaženje pacijenta
    this.vrstaPregledaForm = this.fb.group({
      tip: ['Inicijalni', Validators.required], // 'Inicijalni' ili 'Kontrola'
      pretragaPacijenta: [''], // Koristi se ako je kontrola pa tražimo postojećeg
    });

    // Korak 2: Podaci o pacijentu (Inicijalni karton - popunjava se SAMO ako je tip 'Inicijalni')
    this.pacijentForm = this.fb.group({
      imePrezime: ['', Validators.required],
      jmbg: ['', [Validators.required, Validators.pattern('^[0-9]{13}$')]],
      godiste: ['', Validators.required],
      porodicnaAnamneza: [''],
      opsteNapomene: [''],
    });

    // Korak 3: Medicinski parametri, slike i nalaz (Zajedničko za oba tipa)
    this.pregledForm = this.fb.group({
      iopLevo: ['', [Validators.required, Validators.min(0)]],
      iopDesno: ['', [Validators.required, Validators.min(0)]],
      cdrLevo: ['', Validators.required],
      cdrDesno: ['', Validators.required],
      klinickiNalaz: ['', Validators.required],
    });

    // Logika: Ako lekar promeni na "Kontrola", polja za novog pacijenta prestaju biti obavezna
    this.vrstaPregledaForm.get('tip')?.valueChanges.subscribe((tip) => {
      if (tip === 'Kontrola') {
        this.pacijentForm.disable(); // Isključujemo ceo korak za karton jer pacijent već postoji
      } else {
        this.pacijentForm.enable(); // Uključujemo ponovo ako se vrati na Inicijalni
      }
    });
  }

  // Simulacija izbora fajla za sliku oka
  onFileSelected(event: any, oko: 'levo' | 'desno') {
    const file = event.target.files[0];
    if (file) {
      if (oko === 'levo') this.slikaLevoIme = file.name;
      if (oko === 'desno') this.slikaDesnoIme = file.name;
    }
  }

  // Finalno čuvanje u bazu
  sacuvajPregled() {
    if (this.pregledForm.valid) {
      console.log('Podaci sačuvani:', {
        tip: this.vrstaPregledaForm.value.tip,
        pacijent: this.pacijentForm.value,
        pregled: this.pregledForm.value,
      });

      // Nakon čuvanja, vraćamo lekara na glavnu tablu
      this.router.navigate(['/dashboard']);
    }
  }
}
