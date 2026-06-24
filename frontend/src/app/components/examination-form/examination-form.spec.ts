import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExaminationForm } from './examination-form';

describe('ExaminationForm', () => {
  let component: ExaminationForm;
  let fixture: ComponentFixture<ExaminationForm>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ExaminationForm],
    }).compileComponents();

    fixture = TestBed.createComponent(ExaminationForm);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
